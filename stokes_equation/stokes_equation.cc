/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2007 - 2017 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Martin Kronbichler, Uppsala University,
 *          Wolfgang Bangerth, Texas A&M University 2007, 2008
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include "demo_constants.h"
#include "demo_app.h"

using namespace dealii;

void                    *db             = NULL;
hcq_handle_t            hcq_from_driver = HCQ_INVALID_HANDLE;
hcq_handle_t            hcq_to_driver   = HCQ_INVALID_HANDLE;
const char              *app            = "appB";

namespace EquationData
{
  const double eta = 1;
  const double kappa = 1e-6;
  const double beta = 10;
  const double density = 1;


  template <int dim>
  class TemperatureInitialValues : public Function<dim>
  {
  public:
    TemperatureInitialValues () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };


  template <int dim>
  double
  TemperatureInitialValues<dim>::value (const Point<dim> &,
                                        const unsigned int) const
  {
    return 0;
  }


  template <int dim>
  void
  TemperatureInitialValues<dim>::vector_value (const Point<dim> &p,
                                               Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = TemperatureInitialValues<dim>::value (p, c);
  }


  template <int dim>
  class TemperatureRightHandSide : public Function<dim>
  {
  public:
    TemperatureRightHandSide () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };


  template <int dim>
  double
  TemperatureRightHandSide<dim>::value (const Point<dim>  &p,
                                        const unsigned int component) const
  {
    (void) component;
    Assert (component == 0,
            ExcMessage ("Invalid operation for a scalar function."));

    Assert (dim==2, ExcNotImplemented());

    // The value should come from DTK. To current value is just for testing
    if ((p[0] < 8.) && (p[0] > 4.) && (p[1] < 1.2))
      return 1.;
    else
      return 0.;
  }


  template <int dim>
  void
  TemperatureRightHandSide<dim>::vector_value (const Point<dim> &p,
                                               Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = TemperatureRightHandSide<dim>::value (p, c);
  }
}




namespace LinearSolvers
{


  template <class MatrixType, class PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix (const MatrixType     &m,
                   const PreconditionerType &preconditioner);


    template <typename VectorType>
    void vmult (VectorType       &dst,
                const VectorType &src) const;

  private:
    const SmartPointer<const MatrixType> matrix;
    const PreconditionerType &preconditioner;
  };


  template <class MatrixType, class PreconditionerType>
  InverseMatrix<MatrixType,PreconditionerType>::InverseMatrix
  (const MatrixType         &m,
   const PreconditionerType &preconditioner)
    :
    matrix (&m),
    preconditioner (preconditioner)
  {}



  template <class MatrixType, class PreconditionerType>
  template <typename VectorType>
  void
  InverseMatrix<MatrixType,PreconditionerType>::vmult
  (VectorType       &dst,
   const VectorType &src) const
  {
    SolverControl solver_control (src.size(), 1e-7*src.l2_norm());
    SolverCG<VectorType> cg (solver_control);

    dst = 0;

    try
      {
        cg.solve (*matrix, dst, src, preconditioner);
      }
    catch (std::exception &e)
      {
        Assert (false, ExcMessage(e.what()));
      }
  }


  template <class PreconditionerTypeA, class PreconditionerTypeMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner (
      const TrilinosWrappers::BlockSparseMatrix     &S,
      const InverseMatrix<TrilinosWrappers::SparseMatrix,
      PreconditionerTypeMp>     &Mpinv,
      const PreconditionerTypeA &Apreconditioner);

    void vmult (TrilinosWrappers::MPI::BlockVector       &dst,
                const TrilinosWrappers::MPI::BlockVector &src) const;

  private:
    const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> stokes_matrix;
    const SmartPointer<const InverseMatrix<TrilinosWrappers::SparseMatrix,
          PreconditionerTypeMp > > m_inverse;
    const PreconditionerTypeA &a_preconditioner;

    mutable TrilinosWrappers::MPI::Vector tmp;
  };



  template <class PreconditionerTypeA, class PreconditionerTypeMp>
  BlockSchurPreconditioner<PreconditionerTypeA, PreconditionerTypeMp>::
  BlockSchurPreconditioner(const TrilinosWrappers::BlockSparseMatrix &S,
                           const InverseMatrix<TrilinosWrappers::SparseMatrix,
                           PreconditionerTypeMp>                     &Mpinv,
                           const PreconditionerTypeA                 &Apreconditioner)
    :
    stokes_matrix           (&S),
    m_inverse               (&Mpinv),
    a_preconditioner        (Apreconditioner),
    tmp                     (complete_index_set(stokes_matrix->block(1,1).m()))
  {}


  template <class PreconditionerTypeA, class PreconditionerTypeMp>
  void
  BlockSchurPreconditioner<PreconditionerTypeA, PreconditionerTypeMp>::vmult
  (TrilinosWrappers::MPI::BlockVector       &dst,
   const TrilinosWrappers::MPI::BlockVector &src) const
  {
    a_preconditioner.vmult (dst.block(0), src.block(0));
    stokes_matrix->block(1,0).residual(tmp, dst.block(0), src.block(1));
    tmp *= -1;
    m_inverse->vmult (dst.block(1), tmp);
  }
}




template <int dim>
class BoussinesqFlowProblem
{
public:
  BoussinesqFlowProblem ();
  void run ();

private:
  void setup_dofs ();
  void assemble_stokes_preconditioner ();
  void build_stokes_preconditioner ();
  void assemble_stokes_system ();
  void assemble_temperature_system (const double maximal_velocity);
  void assemble_temperature_matrix ();
  double get_maximal_velocity () const;
  std::pair<double,double> get_extrapolated_temperature_range () const;
  void solve ();
  void output_results () const;

  double
  compute_viscosity(const std::vector<double>          &old_temperature,
                    const std::vector<double>          &old_old_temperature,
                    const std::vector<Tensor<1,dim> >  &old_temperature_grads,
                    const std::vector<Tensor<1,dim> >  &old_old_temperature_grads,
                    const std::vector<double>          &old_temperature_laplacians,
                    const std::vector<double>          &old_old_temperature_laplacians,
                    const std::vector<Tensor<1,dim> >  &old_velocity_values,
                    const std::vector<Tensor<1,dim> >  &old_old_velocity_values,
                    const std::vector<double>          &gamma_values,
                    const double                        global_u_infty,
                    const double                        global_T_variation,
                    const double                        cell_diameter) const;


  Triangulation<dim>                  triangulation;
  double                              global_Omega_diameter;

  const unsigned int                  stokes_degree;
  FESystem<dim>                       stokes_fe;
  DoFHandler<dim>                     stokes_dof_handler;
  ConstraintMatrix                    stokes_constraints;

  std::vector<IndexSet>               stokes_partitioning;
  TrilinosWrappers::BlockSparseMatrix stokes_matrix;
  TrilinosWrappers::BlockSparseMatrix stokes_preconditioner_matrix;

  TrilinosWrappers::MPI::BlockVector  stokes_solution;
  TrilinosWrappers::MPI::BlockVector  old_stokes_solution;
  TrilinosWrappers::MPI::BlockVector  stokes_rhs;


  const unsigned int                  temperature_degree;
  FE_Q<dim>                           temperature_fe;
  DoFHandler<dim>                     temperature_dof_handler;
  ConstraintMatrix                    temperature_constraints;

  TrilinosWrappers::SparseMatrix      temperature_mass_matrix;
  TrilinosWrappers::SparseMatrix      temperature_stiffness_matrix;
  TrilinosWrappers::SparseMatrix      temperature_matrix;

  TrilinosWrappers::MPI::Vector       temperature_solution;
  TrilinosWrappers::MPI::Vector       old_temperature_solution;
  TrilinosWrappers::MPI::Vector       old_old_temperature_solution;
  TrilinosWrappers::MPI::Vector       temperature_rhs;


  double                              time_step;
  double                              old_time_step;
  unsigned int                        timestep_number;

  std::shared_ptr<TrilinosWrappers::PreconditionAMG> Amg_preconditioner;
  std::shared_ptr<TrilinosWrappers::PreconditionIC>  Mp_preconditioner;

  bool                                rebuild_stokes_matrix;
  bool                                rebuild_temperature_matrices;
  bool                                rebuild_stokes_preconditioner;
};



template <int dim>
BoussinesqFlowProblem<dim>::BoussinesqFlowProblem ()
  :
  triangulation (Triangulation<dim>::maximum_smoothing),
  global_Omega_diameter (std::numeric_limits<double>::quiet_NaN()),
  stokes_degree (1),
  stokes_fe (FE_Q<dim>(stokes_degree+1), dim,
             FE_Q<dim>(stokes_degree), 1),
  stokes_dof_handler (triangulation),

  temperature_degree (2),
  temperature_fe (temperature_degree),
  temperature_dof_handler (triangulation),

  time_step (0),
  old_time_step (0),
  timestep_number (0),
  rebuild_stokes_matrix (true),
  rebuild_temperature_matrices (true),
  rebuild_stokes_preconditioner (true)
{}




template <int dim>
double BoussinesqFlowProblem<dim>::get_maximal_velocity () const
{
  const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                           stokes_degree+1);
  const unsigned int n_q_points = quadrature_formula.size();

  FEValues<dim> fe_values (stokes_fe, quadrature_formula, update_values);
  std::vector<Tensor<1,dim> > velocity_values(n_q_points);
  double max_velocity = 0;

  const FEValuesExtractors::Vector velocities (0);

  typename DoFHandler<dim>::active_cell_iterator
  cell = stokes_dof_handler.begin_active(),
  endc = stokes_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      fe_values[velocities].get_function_values (stokes_solution,
                                                 velocity_values);

      for (unsigned int q=0; q<n_q_points; ++q)
        max_velocity = std::max (max_velocity, velocity_values[q].norm());
    }

  return max_velocity;
}





template <int dim>
std::pair<double,double>
BoussinesqFlowProblem<dim>::get_extrapolated_temperature_range () const
{
  const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                           temperature_degree);
  const unsigned int n_q_points = quadrature_formula.size();

  FEValues<dim> fe_values (temperature_fe, quadrature_formula,
                           update_values);
  std::vector<double> old_temperature_values(n_q_points);
  std::vector<double> old_old_temperature_values(n_q_points);

  if (timestep_number != 0)
    {
      double min_temperature = std::numeric_limits<double>::max(),
             max_temperature = -std::numeric_limits<double>::max();

      typename DoFHandler<dim>::active_cell_iterator
      cell = temperature_dof_handler.begin_active(),
      endc = temperature_dof_handler.end();
      for (; cell!=endc; ++cell)
        {
          fe_values.reinit (cell);
          fe_values.get_function_values (old_temperature_solution,
                                         old_temperature_values);
          fe_values.get_function_values (old_old_temperature_solution,
                                         old_old_temperature_values);

          for (unsigned int q=0; q<n_q_points; ++q)
            {
              const double temperature =
                (1. + time_step/old_time_step) * old_temperature_values[q]-
                time_step/old_time_step * old_old_temperature_values[q];

              min_temperature = std::min (min_temperature, temperature);
              max_temperature = std::max (max_temperature, temperature);
            }
        }

      return std::make_pair(min_temperature, max_temperature);
    }
  else
    {
      double min_temperature = std::numeric_limits<double>::max(),
             max_temperature = -std::numeric_limits<double>::max();

      typename DoFHandler<dim>::active_cell_iterator
      cell = temperature_dof_handler.begin_active(),
      endc = temperature_dof_handler.end();
      for (; cell!=endc; ++cell)
        {
          fe_values.reinit (cell);
          fe_values.get_function_values (old_temperature_solution,
                                         old_temperature_values);

          for (unsigned int q=0; q<n_q_points; ++q)
            {
              const double temperature = old_temperature_values[q];

              min_temperature = std::min (min_temperature, temperature);
              max_temperature = std::max (max_temperature, temperature);
            }
        }

      return std::make_pair(min_temperature, max_temperature);
    }
}




template <int dim>
double
BoussinesqFlowProblem<dim>::
compute_viscosity (const std::vector<double>          &old_temperature,
                   const std::vector<double>          &old_old_temperature,
                   const std::vector<Tensor<1,dim> >  &old_temperature_grads,
                   const std::vector<Tensor<1,dim> >  &old_old_temperature_grads,
                   const std::vector<double>          &old_temperature_laplacians,
                   const std::vector<double>          &old_old_temperature_laplacians,
                   const std::vector<Tensor<1,dim> >  &old_velocity_values,
                   const std::vector<Tensor<1,dim> >  &old_old_velocity_values,
                   const std::vector<double>          &gamma_values,
                   const double                        global_u_infty,
                   const double                        global_T_variation,
                   const double                        cell_diameter) const
{
  const double beta = 0.017 * dim;
  const double alpha = 1;

  if (global_u_infty == 0)
    return 5e-3 * cell_diameter;

  const unsigned int n_q_points = old_temperature.size();

  double max_residual = 0;
  double max_velocity = 0;

  for (unsigned int q=0; q < n_q_points; ++q)
    {
      const Tensor<1,dim> u = (old_velocity_values[q] +
                               old_old_velocity_values[q]) / 2;

      const double dT_dt = (old_temperature[q] - old_old_temperature[q])
                           / old_time_step;
      const double u_grad_T = u * (old_temperature_grads[q] +
                                   old_old_temperature_grads[q]) / 2;

      const double kappa_Delta_T = EquationData::kappa
                                   * (old_temperature_laplacians[q] +
                                      old_old_temperature_laplacians[q]) / 2;

      const double residual
        = std::abs((dT_dt + u_grad_T - kappa_Delta_T - gamma_values[q]) *
                   std::pow((old_temperature[q]+old_old_temperature[q]) / 2,
                            alpha-1.));

      max_residual = std::max (residual,        max_residual);
      max_velocity = std::max (std::sqrt (u*u), max_velocity);
    }

  const double c_R = std::pow (2., (4.-2*alpha)/dim);
  const double global_scaling = c_R * global_u_infty * global_T_variation *
                                std::pow(global_Omega_diameter, alpha - 2.);

  return (beta *
          max_velocity *
          std::min (cell_diameter,
                    std::pow(cell_diameter,alpha) *
                    max_residual / global_scaling));
}



template <int dim>
void BoussinesqFlowProblem<dim>::setup_dofs ()
{
  std::vector<unsigned int> stokes_sub_blocks (dim+1,0);
  stokes_sub_blocks[dim] = 1;

  {
    stokes_dof_handler.distribute_dofs (stokes_fe);
    DoFRenumbering::component_wise (stokes_dof_handler, stokes_sub_blocks);

    stokes_constraints.clear ();
    DoFTools::make_hanging_node_constraints (stokes_dof_handler,
                                             stokes_constraints);
    std::set<types::boundary_id> no_normal_flux_boundaries;
    no_normal_flux_boundaries.insert (0);
    VectorTools::compute_no_normal_flux_constraints (stokes_dof_handler, 0,
                                                     no_normal_flux_boundaries,
                                                     stokes_constraints);
    stokes_constraints.close ();
  }
  {
    temperature_dof_handler.distribute_dofs (temperature_fe);

    temperature_constraints.clear ();
    DoFTools::make_hanging_node_constraints (temperature_dof_handler,
                                             temperature_constraints);
    temperature_constraints.close ();
  }

  std::vector<types::global_dof_index> stokes_dofs_per_block (2);
//  Hobbes::vector<types::global_dof_index> stokes_dofs_per_block (2);
  DoFTools::count_dofs_per_block (stokes_dof_handler, stokes_dofs_per_block,
                                  stokes_sub_blocks);

  const unsigned int n_u = stokes_dofs_per_block[0],
                     n_p = stokes_dofs_per_block[1],
                     n_T = temperature_dof_handler.n_dofs();

//  stokes_dofs_per_block._export(app);

  std::cout << "Number of active cells: "
            << triangulation.n_active_cells()
            << " (on "
            << triangulation.n_levels()
            << " levels)"
            << std::endl
            << "Number of degrees of freedom: "
            << n_u + n_p + n_T
            << " (" << n_u << '+' << n_p << '+'<< n_T <<')'
            << std::endl
            << std::endl;

  stokes_partitioning.resize (2);
  stokes_partitioning[0] = complete_index_set (n_u);
  stokes_partitioning[1] = complete_index_set (n_p);
  {
    stokes_matrix.clear ();

    BlockDynamicSparsityPattern dsp (2,2);

    dsp.block(0,0).reinit (n_u, n_u);
    dsp.block(0,1).reinit (n_u, n_p);
    dsp.block(1,0).reinit (n_p, n_u);
    dsp.block(1,1).reinit (n_p, n_p);

    dsp.collect_sizes ();

    Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);

    for (unsigned int c=0; c<dim+1; ++c)
      for (unsigned int d=0; d<dim+1; ++d)
        if (! ((c==dim) && (d==dim)))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern (stokes_dof_handler, coupling, dsp,
                                     stokes_constraints, false);

    stokes_matrix.reinit (dsp);
  }

  {
    Amg_preconditioner.reset ();
    Mp_preconditioner.reset ();
    stokes_preconditioner_matrix.clear ();

    BlockDynamicSparsityPattern dsp (2,2);

    dsp.block(0,0).reinit (n_u, n_u);
    dsp.block(0,1).reinit (n_u, n_p);
    dsp.block(1,0).reinit (n_p, n_u);
    dsp.block(1,1).reinit (n_p, n_p);

    dsp.collect_sizes ();

    Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);
    for (unsigned int c=0; c<dim+1; ++c)
      for (unsigned int d=0; d<dim+1; ++d)
        if (c == d)
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern (stokes_dof_handler, coupling, dsp,
                                     stokes_constraints, false);

    stokes_preconditioner_matrix.reinit (dsp);
  }

  {
    temperature_mass_matrix.clear ();
    temperature_stiffness_matrix.clear ();
    temperature_matrix.clear ();

    DynamicSparsityPattern dsp (n_T, n_T);
    DoFTools::make_sparsity_pattern (temperature_dof_handler, dsp,
                                     temperature_constraints, false);

    temperature_matrix.reinit (dsp);
    temperature_mass_matrix.reinit (temperature_matrix);
    temperature_stiffness_matrix.reinit (temperature_matrix);
  }

  IndexSet temperature_partitioning = complete_index_set (n_T);
  stokes_solution.reinit (stokes_partitioning, MPI_COMM_WORLD);
  old_stokes_solution.reinit (stokes_partitioning, MPI_COMM_WORLD);
  stokes_rhs.reinit (stokes_partitioning, MPI_COMM_WORLD);

  temperature_solution.reinit (temperature_partitioning, MPI_COMM_WORLD);
  old_temperature_solution.reinit (temperature_partitioning, MPI_COMM_WORLD);
  old_old_temperature_solution.reinit (temperature_partitioning, MPI_COMM_WORLD);

  temperature_rhs.reinit (temperature_partitioning, MPI_COMM_WORLD);
}



template <int dim>
void
BoussinesqFlowProblem<dim>::assemble_stokes_preconditioner ()
{
  stokes_preconditioner_matrix = 0;

  const QGauss<dim> quadrature_formula(stokes_degree+2);
  FEValues<dim>     stokes_fe_values (stokes_fe, quadrature_formula,
                                      update_JxW_values |
                                      update_values |
                                      update_gradients);

  const unsigned int   dofs_per_cell   = stokes_fe.dofs_per_cell;
  const unsigned int   n_q_points      = quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  std::vector<Tensor<2,dim> > grad_phi_u (dofs_per_cell);
  std::vector<double>         phi_p      (dofs_per_cell);

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  typename DoFHandler<dim>::active_cell_iterator
  cell = stokes_dof_handler.begin_active(),
  endc = stokes_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      stokes_fe_values.reinit (cell);
      local_matrix = 0;

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
              grad_phi_u[k] = stokes_fe_values[velocities].gradient(k,q);
              phi_p[k]      = stokes_fe_values[pressure].value (k, q);
            }

          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              local_matrix(i,j) += (EquationData::eta *
                                    scalar_product (grad_phi_u[i], grad_phi_u[j])
                                    +
                                    (1./EquationData::eta) *
                                    phi_p[i] * phi_p[j])
                                   * stokes_fe_values.JxW(q);
        }

      cell->get_dof_indices (local_dof_indices);
      stokes_constraints.distribute_local_to_global (local_matrix,
                                                     local_dof_indices,
                                                     stokes_preconditioner_matrix);
    }
}



template <int dim>
void
BoussinesqFlowProblem<dim>::build_stokes_preconditioner ()
{
  if (rebuild_stokes_preconditioner == false)
    return;

  std::cout << "   Rebuilding Stokes preconditioner..." << std::flush;

  assemble_stokes_preconditioner ();

  Amg_preconditioner = std::shared_ptr<TrilinosWrappers::PreconditionAMG>
                       (new TrilinosWrappers::PreconditionAMG());

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector velocity_components(0);
  DoFTools::extract_constant_modes (stokes_dof_handler,
                                    stokes_fe.component_mask(velocity_components),
                                    constant_modes);
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.constant_modes = constant_modes;

  amg_data.elliptic = true;
  amg_data.higher_order_elements = true;
  amg_data.smoother_sweeps = 2;
  amg_data.aggregation_threshold = 0.02;
  Amg_preconditioner->initialize(stokes_preconditioner_matrix.block(0,0),
                                 amg_data);

  Mp_preconditioner = std::shared_ptr<TrilinosWrappers::PreconditionIC>
                      (new TrilinosWrappers::PreconditionIC());
  Mp_preconditioner->initialize(stokes_preconditioner_matrix.block(1,1));

  std::cout << std::endl;

  rebuild_stokes_preconditioner = false;
}



template <int dim>
void BoussinesqFlowProblem<dim>::assemble_stokes_system ()
{
  std::cout << "   Assembling..." << std::flush;

  if (rebuild_stokes_matrix == true)
    stokes_matrix=0;

  stokes_rhs=0;

  const QGauss<dim> quadrature_formula (stokes_degree+2);
  FEValues<dim>     stokes_fe_values (stokes_fe, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      (rebuild_stokes_matrix == true
                                       ?
                                       update_gradients
                                       :
                                       UpdateFlags(0)));

  FEValues<dim>     temperature_fe_values (temperature_fe, quadrature_formula,
                                           update_values);

  const unsigned int   dofs_per_cell   = stokes_fe.dofs_per_cell;
  const unsigned int   n_q_points      = quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       local_rhs    (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  std::vector<double>               old_temperature_values(n_q_points);

  std::vector<Tensor<1,dim> >          phi_u       (dofs_per_cell);
  std::vector<SymmetricTensor<2,dim> > grads_phi_u (dofs_per_cell);
  std::vector<double>                  div_phi_u   (dofs_per_cell);
  std::vector<double>                  phi_p       (dofs_per_cell);

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  typename DoFHandler<dim>::active_cell_iterator
  cell = stokes_dof_handler.begin_active(),
  endc = stokes_dof_handler.end();
  typename DoFHandler<dim>::active_cell_iterator
  temperature_cell = temperature_dof_handler.begin_active();

  for (; cell!=endc; ++cell, ++temperature_cell)
    {
      stokes_fe_values.reinit (cell);
      temperature_fe_values.reinit (temperature_cell);

      local_matrix = 0;
      local_rhs = 0;

      temperature_fe_values.get_function_values (old_temperature_solution,
                                                 old_temperature_values);

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          const double old_temperature = old_temperature_values[q];

          for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
              phi_u[k] = stokes_fe_values[velocities].value (k,q);
              if (rebuild_stokes_matrix)
                {
                  grads_phi_u[k] = stokes_fe_values[velocities].symmetric_gradient(k,q);
                  div_phi_u[k]   = stokes_fe_values[velocities].divergence (k, q);
                  phi_p[k]       = stokes_fe_values[pressure].value (k, q);
                }
            }

          if (rebuild_stokes_matrix)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                local_matrix(i,j) += (EquationData::eta * 2 *
                                      (grads_phi_u[i] * grads_phi_u[j])
                                      - div_phi_u[i] * phi_p[j]
                                      - phi_p[i] * div_phi_u[j])
                                     * stokes_fe_values.JxW(q);

          const Point<dim> gravity = -( (dim == 2) ? (Point<dim> (0,1)) :
                                        (Point<dim> (0,0,1)) );
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            local_rhs(i) += (-EquationData::density *
                             EquationData::beta *
                             gravity * phi_u[i] * old_temperature)*
                            stokes_fe_values.JxW(q);
        }

      cell->get_dof_indices (local_dof_indices);

      if (rebuild_stokes_matrix == true)
        stokes_constraints.distribute_local_to_global (local_matrix,
                                                       local_rhs,
                                                       local_dof_indices,
                                                       stokes_matrix,
                                                       stokes_rhs);
      else
        stokes_constraints.distribute_local_to_global (local_rhs,
                                                       local_dof_indices,
                                                       stokes_rhs);
    }

  rebuild_stokes_matrix = false;

  std::cout << std::endl;
}




template <int dim>
void BoussinesqFlowProblem<dim>::assemble_temperature_matrix ()
{
  if (rebuild_temperature_matrices == false)
    return;

  temperature_mass_matrix = 0;
  temperature_stiffness_matrix = 0;

  QGauss<dim>   quadrature_formula (temperature_degree+2);
  FEValues<dim> temperature_fe_values (temperature_fe, quadrature_formula,
                                       update_values    | update_gradients |
                                       update_JxW_values);

  const unsigned int   dofs_per_cell   = temperature_fe.dofs_per_cell;
  const unsigned int   n_q_points      = quadrature_formula.size();

  FullMatrix<double>   local_mass_matrix (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>   local_stiffness_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

//  std::vector<double>         phi_T       (dofs_per_cell);
  Hobbes::vector<double>         phi_T       (dofs_per_cell);
  phi_T._export (app);
  std::vector<Tensor<1,dim> > grad_phi_T  (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator

  cell = temperature_dof_handler.begin_active(),
  endc = temperature_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      local_mass_matrix = 0;
      local_stiffness_matrix = 0;

      temperature_fe_values.reinit (cell);

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
              grad_phi_T[k] = temperature_fe_values.shape_grad (k,q);
              phi_T[k]      = temperature_fe_values.shape_value (k, q);
            }

          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              {
                local_mass_matrix(i,j)
                += (phi_T[i] * phi_T[j]
                    *
                    temperature_fe_values.JxW(q));
                local_stiffness_matrix(i,j)
                += (EquationData::kappa * grad_phi_T[i] * grad_phi_T[j]
                    *
                    temperature_fe_values.JxW(q));
              }
        }

      cell->get_dof_indices (local_dof_indices);

      temperature_constraints.distribute_local_to_global (local_mass_matrix,
                                                          local_dof_indices,
                                                          temperature_mass_matrix);
      temperature_constraints.distribute_local_to_global (local_stiffness_matrix,
                                                          local_dof_indices,
                                                          temperature_stiffness_matrix);
    }

  rebuild_temperature_matrices = false;
}



template <int dim>
void BoussinesqFlowProblem<dim>::
assemble_temperature_system (const double maximal_velocity)
{
  const bool use_bdf2_scheme = (timestep_number != 0);

  if (use_bdf2_scheme == true)
    {
      temperature_matrix.copy_from (temperature_mass_matrix);
      temperature_matrix *= (2*time_step + old_time_step) /
                            (time_step + old_time_step);
      temperature_matrix.add (time_step, temperature_stiffness_matrix);
    }
  else
    {
      temperature_matrix.copy_from (temperature_mass_matrix);
      temperature_matrix.add (time_step, temperature_stiffness_matrix);
    }

  temperature_rhs = 0;

  const QGauss<dim> quadrature_formula(temperature_degree+2);
  FEValues<dim>     temperature_fe_values (temperature_fe, quadrature_formula,
                                           update_values    |
                                           update_gradients |
                                           update_hessians  |
                                           update_quadrature_points  |
                                           update_JxW_values);
  FEValues<dim>     stokes_fe_values (stokes_fe, quadrature_formula,
                                      update_values);

  const unsigned int   dofs_per_cell   = temperature_fe.dofs_per_cell;
  const unsigned int   n_q_points      = quadrature_formula.size();

  Vector<double>       local_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  std::vector<Tensor<1,dim> > old_velocity_values (n_q_points);
  std::vector<Tensor<1,dim> > old_old_velocity_values (n_q_points);
  std::vector<double>         old_temperature_values (n_q_points);
  std::vector<double>         old_old_temperature_values(n_q_points);
  std::vector<Tensor<1,dim> > old_temperature_grads(n_q_points);
  std::vector<Tensor<1,dim> > old_old_temperature_grads(n_q_points);
  std::vector<double>         old_temperature_laplacians(n_q_points);
  std::vector<double>         old_old_temperature_laplacians(n_q_points);

  EquationData::TemperatureRightHandSide<dim>  temperature_right_hand_side;
  std::vector<double> gamma_values (n_q_points);

  std::vector<double>         phi_T      (dofs_per_cell);
  std::vector<Tensor<1,dim> > grad_phi_T (dofs_per_cell);

  const std::pair<double,double>
  global_T_range = get_extrapolated_temperature_range();

  const FEValuesExtractors::Vector velocities (0);

  typename DoFHandler<dim>::active_cell_iterator
  cell = temperature_dof_handler.begin_active(),
  endc = temperature_dof_handler.end();
  typename DoFHandler<dim>::active_cell_iterator
  stokes_cell = stokes_dof_handler.begin_active();

  for (; cell!=endc; ++cell, ++stokes_cell)
    {
      local_rhs = 0;

      temperature_fe_values.reinit (cell);
      stokes_fe_values.reinit (stokes_cell);

      temperature_fe_values.get_function_values (old_temperature_solution,
                                                 old_temperature_values);
      temperature_fe_values.get_function_values (old_old_temperature_solution,
                                                 old_old_temperature_values);

      temperature_fe_values.get_function_gradients (old_temperature_solution,
                                                    old_temperature_grads);
      temperature_fe_values.get_function_gradients (old_old_temperature_solution,
                                                    old_old_temperature_grads);

      temperature_fe_values.get_function_laplacians (old_temperature_solution,
                                                     old_temperature_laplacians);
      temperature_fe_values.get_function_laplacians (old_old_temperature_solution,
                                                     old_old_temperature_laplacians);

      temperature_right_hand_side.value_list (temperature_fe_values.get_quadrature_points(),
                                              gamma_values);

      stokes_fe_values[velocities].get_function_values (stokes_solution,
                                                        old_velocity_values);
      stokes_fe_values[velocities].get_function_values (old_stokes_solution,
                                                        old_old_velocity_values);

      const double nu
        = compute_viscosity (old_temperature_values,
                             old_old_temperature_values,
                             old_temperature_grads,
                             old_old_temperature_grads,
                             old_temperature_laplacians,
                             old_old_temperature_laplacians,
                             old_velocity_values,
                             old_old_velocity_values,
                             gamma_values,
                             maximal_velocity,
                             global_T_range.second - global_T_range.first,
                             cell->diameter());

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
              grad_phi_T[k] = temperature_fe_values.shape_grad (k,q);
              phi_T[k]      = temperature_fe_values.shape_value (k, q);
            }

          const double T_term_for_rhs
            = (use_bdf2_scheme ?
               (old_temperature_values[q] *
                (1 + time_step/old_time_step)
                -
                old_old_temperature_values[q] *
                (time_step * time_step) /
                (old_time_step * (time_step + old_time_step)))
               :
               old_temperature_values[q]);

          const Tensor<1,dim> ext_grad_T
            = (use_bdf2_scheme ?
               (old_temperature_grads[q] *
                (1 + time_step/old_time_step)
                -
                old_old_temperature_grads[q] *
                time_step/old_time_step)
               :
               old_temperature_grads[q]);

          const Tensor<1,dim> extrapolated_u
            = (use_bdf2_scheme ?
               (old_velocity_values[q] *
                (1 + time_step/old_time_step)
                -
                old_old_velocity_values[q] *
                time_step/old_time_step)
               :
               old_velocity_values[q]);

          for (unsigned int i=0; i<dofs_per_cell; ++i)
            local_rhs(i) += (T_term_for_rhs * phi_T[i]
                             -
                             time_step *
                             extrapolated_u * ext_grad_T * phi_T[i]
                             -
                             time_step *
                             nu * ext_grad_T * grad_phi_T[i]
                             +
                             time_step *
                             gamma_values[q] * phi_T[i])
                            *
                            temperature_fe_values.JxW(q);
        }

      cell->get_dof_indices (local_dof_indices);
      temperature_constraints.distribute_local_to_global (local_rhs,
                                                          local_dof_indices,
                                                          temperature_rhs);
    }
}




template <int dim>
void BoussinesqFlowProblem<dim>::solve ()
{
  std::cout << "   Solving..." << std::endl;

  {
    const LinearSolvers::InverseMatrix<TrilinosWrappers::SparseMatrix,
          TrilinosWrappers::PreconditionIC>
          mp_inverse (stokes_preconditioner_matrix.block(1,1), *Mp_preconditioner);

    const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
          TrilinosWrappers::PreconditionIC>
          preconditioner (stokes_matrix, mp_inverse, *Amg_preconditioner);

    SolverControl solver_control (stokes_matrix.m(),
                                  1e-6*stokes_rhs.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::BlockVector>
    gmres (solver_control,
           SolverGMRES<TrilinosWrappers::MPI::BlockVector >::AdditionalData(100));

    for (unsigned int i=0; i<stokes_solution.size(); ++i)
      if (stokes_constraints.is_constrained(i))
        stokes_solution(i) = 0;

    gmres.solve(stokes_matrix, stokes_solution, stokes_rhs, preconditioner);

    stokes_constraints.distribute (stokes_solution);

    std::cout << "   "
              << solver_control.last_step()
              << " GMRES iterations for Stokes subsystem."
              << std::endl;
  }

  old_time_step = time_step;
  const double maximal_velocity = get_maximal_velocity();

  if (maximal_velocity >= 0.01)
    time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
                temperature_degree *
                GridTools::minimal_cell_diameter(triangulation) /
                maximal_velocity;
  else
    time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
                temperature_degree *
                GridTools::minimal_cell_diameter(triangulation) /
                .01;
  double const heat_equation_time_step = 1./500.;
  time_step = std::min(time_step, heat_equation_time_step);

  std::cout << "   " << "Time step: " << time_step
            << std::endl;

  temperature_solution = old_temperature_solution;

  assemble_temperature_system (maximal_velocity);
  {

    SolverControl solver_control (temperature_matrix.m(),
                                  1e-8*temperature_rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionIC preconditioner;
    preconditioner.initialize (temperature_matrix);

    cg.solve (temperature_matrix, temperature_solution,
              temperature_rhs, preconditioner);

    temperature_constraints.distribute (temperature_solution);

    std::cout << "   "
              << solver_control.last_step()
              << " CG iterations for temperature."
              << std::endl;

    double min_temperature = temperature_solution(0),
           max_temperature = temperature_solution(0);
    for (unsigned int i=0; i<temperature_solution.size(); ++i)
      {
        min_temperature = std::min<double> (min_temperature,
                                            temperature_solution(i));
        max_temperature = std::max<double> (max_temperature,
                                            temperature_solution(i));
      }

    std::cout << "   Temperature range: "
              << min_temperature << ' ' << max_temperature
              << std::endl;
  }
}



template <int dim>
void BoussinesqFlowProblem<dim>::output_results ()  const
{
  if (timestep_number % 10 != 0)
    return;

  std::vector<std::string> stokes_names (dim, "velocity");
  stokes_names.push_back ("p");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  stokes_component_interpretation
  (dim+1, DataComponentInterpretation::component_is_scalar);
  for (unsigned int i=0; i<dim; ++i)
    stokes_component_interpretation[i]
      = DataComponentInterpretation::component_is_part_of_vector;

  DataOut<dim> data_out;
  data_out.add_data_vector (stokes_dof_handler, stokes_solution,
                            stokes_names, stokes_component_interpretation);
  data_out.add_data_vector (temperature_dof_handler, temperature_solution,
                            "T");
  data_out.build_patches (std::min(stokes_degree, temperature_degree));

  std::ostringstream filename;
  filename << "solution-" << Utilities::int_to_string(timestep_number, 4) << ".vtk";

  std::ofstream output (filename.str().c_str());
  data_out.write_vtk (output);
}



template <int dim>
void BoussinesqFlowProblem<dim>::run ()
{
  int rc;
  const unsigned int initial_refinement = (dim == 2 ? 5 : 2);

  GridGenerator::hyper_cube (triangulation, 1., 11.);
  global_Omega_diameter = GridTools::diameter (triangulation);

  triangulation.refine_global (initial_refinement);

  setup_dofs();

  VectorTools::project (temperature_dof_handler,
                        temperature_constraints,
                        QGauss<dim>(temperature_degree+2),
                        EquationData::TemperatureInitialValues<dim>(),
                        old_temperature_solution);

  timestep_number           = 0;
  time_step = old_time_step = 0;

  double time = 0;

  do
    {
      rc = _app_wait (hcq_to_driver);
      if (rc == -1)
      {
        fprintf (stderr, "ERROR: _app_wait() failed\n");
      }

      std::cout << "Timestep " << timestep_number
                << ":  t=" << time
                << std::endl;

      assemble_stokes_system ();
      build_stokes_preconditioner ();
      assemble_temperature_matrix ();

      solve ();

      output_results ();

      std::cout << std::endl;

      time += time_step;
      ++timestep_number;

      old_stokes_solution          = stokes_solution;
      old_old_temperature_solution = old_temperature_solution;
      old_temperature_solution     = temperature_solution;

      rc = _app_notify_driver (hcq_from_driver);
      if (rc == -1)
      {
        fprintf (stderr, "ERROR: _app_notify_driver() failed\n");
      }
    }
  while (time <= 100);
}




int main (int argc, char *argv[])
{
  int rc;

  rc = _setup_hobbes (&db, NULL, NULL);
  if (rc != 0)
  {
    fprintf (stderr, "ERROR: _setup_hobbes() failed\n");
    goto exit_fn_on_error;
  }

  /*
   * Setting up 2 comamnd queues for bi-directional commands with the
   * driver.
   */
  rc = _app_handshake (app, &hcq_to_driver, &hcq_from_driver);
  if (rc == -1)
  {
    fprintf (stderr, "ERROR: _app_handshake() failed\n");
    goto exit_fn_on_error;
  }

  // Notifying the driver that we are up...
  rc = _app_notify_driver (hcq_from_driver);
  if (rc == -1)
  {
    fprintf (stderr, "ERROR: _app_notify_driver() failed\n");
    goto exit_fn_on_error;
  }

  try
    {
      using namespace dealii;

      Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
                                                           numbers::invalid_unsigned_int);

      AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)==1,
                  ExcMessage("This program can only be run in serial, use ./step-31"));

      BoussinesqFlowProblem<2> flow_problem;
      flow_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      goto exit_fn_on_error;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      goto exit_fn_on_error;
    }

  rc = _app_wait (hcq_to_driver);
  if (rc == -1)
  {
    fprintf (stderr, "ERROR: _app_wait() failed\n");
    goto exit_fn_on_error;
  }

  hcq_disconnect (hcq_from_driver);
  hdb_detach (db);

  return 0;

 exit_fn_on_error:

  if (hcq_from_driver != HCQ_INVALID_HANDLE)
  {
    hcq_disconnect (hcq_from_driver);
  }

  if (db != NULL)
  {
    hdb_detach (db);
  }

  return 1;
}
