/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2017 by the deal.II authors
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
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */


#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>

#include "demo_constants.h"
#include "demo_app.h"

extern hdb_db_t         hobbes_master_db;
void                    *db             = NULL;
hcq_handle_t            hcq_from_driver = HCQ_INVALID_HANDLE;
hcq_handle_t            hcq_to_driver   = HCQ_INVALID_HANDLE;
const char              *app            = "appA";

using namespace dealii;


template <int dim>
class HeatEquation
{
public:
  HeatEquation();
  void run();

private:
  void build_mesh();
  void setup_system();
  void extract_boundary();
  void solve_time_step();
  void extract_values();
  void output_results() const;

  Triangulation<dim>   triangulation;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;

  ConstraintMatrix     constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       old_solution;
  Vector<double>       system_rhs;

  double               time;
  double               time_step;
  unsigned int         timestep_number;

  const double         theta;

  std::vector<std::pair<Point<dim>, types::global_dof_index>> interior_boundary;
/*
  std::vector<double> interior_values;
  Hobbes::vector<std::pair<Point<dim>, types::global_dof_index>> interior_boundary;
*/
  Hobbes::vector<double> interior_values;
};




template <int dim>
class Coefficient : public Function<dim>
{
public:
  Coefficient()
    :
    Function<dim>()
  {}

  virtual double value (const Point<dim> &p,
                        const unsigned int component = 0) const;
};



template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
                                const unsigned int ) const
{
   return 1.5;
}




template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide ()
    :
    Function<dim>()
  {}

  virtual double value (const Point<dim> &p,
                        const unsigned int component = 0) const;
};



template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int component) const
{
  (void) component;
  Assert (component == 0, ExcIndexRange(component, 0, 1));
  Assert (dim == 2, ExcNotImplemented());

  const double time = this->get_time();
  if ((p[0] >= 4.) && (p[0]<=8.))
    return 1. + time;
  else
    return 0.;
}



template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double value (const Point<dim>  &p,
                        const unsigned int component = 0) const;
};



template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &/*p*/,
                                   const unsigned int component) const
{
  (void) component;
  Assert (component == 0, ExcIndexRange(component, 0, 1));
  return 0;
}



template <int dim>
HeatEquation<dim>::HeatEquation ()
  :
  fe(1),
  dof_handler(triangulation),
  time (0.0),
  time_step(1. / 500),
  timestep_number (0),
  theta(0.5)
{
    interior_values._export(app);
    //interior_boundary._export(app);
}



template <int dim>
void HeatEquation<dim>::build_mesh()
{
  const unsigned int initial_global_refinement = 3;

  // Bottom of the pot
  Triangulation<dim> bottom_triangulation;
  std::vector<unsigned int> repetitions_b(dim, 1);
  repetitions_b[0] = 10;
  Point<dim> bottom_left_b(1., 0.);
  Point<dim> top_right_b(11., 1.);
  GridGenerator::subdivided_hyper_rectangle(bottom_triangulation, repetitions_b,
     bottom_left_b, top_right_b);

  // Right of the pot
  Triangulation<dim> right_triangulation;
  std::vector<unsigned int> repetitions_r(dim, 1);
  repetitions_r[1] = 12;
  Point<dim> bottom_left_r(11., 0.);
  Point<dim> top_right_r(12., 12.);
  GridGenerator::subdivided_hyper_rectangle(right_triangulation, repetitions_r,
     bottom_left_r, top_right_r);

  // Left of the pot
  Triangulation<dim> left_triangulation;
  std::vector<unsigned int> repetitions_l(dim, 1);
  repetitions_l[1] = 12;
  Point<dim> bottom_left_l(0., 0.);
  Point<dim> top_right_l(1., 12.);
  GridGenerator::subdivided_hyper_rectangle(left_triangulation, repetitions_l,
     bottom_left_l, top_right_l);

  // Merge the triangulations
  Triangulation<dim> tmp;
  GridGenerator::merge_triangulations(bottom_triangulation, right_triangulation, tmp);
  GridGenerator::merge_triangulations(tmp, left_triangulation, triangulation);

  // Refine the final mesh
  triangulation.refine_global (initial_global_refinement);
}



template <int dim>
void HeatEquation<dim>::extract_boundary()
{
  Assert(dim == 2, ExcNotImplemented());
  for (auto cell : dof_handler.active_cell_iterators())
  {
    if (cell->at_boundary())
    {
      Point<dim> center = cell->center();
      if ((center[0] > 0.5) && (center[0] < 11.5) && (center[1] > 0.5))
      {
        for (unsigned int i=0; i<GeometryInfo<dim>::faces_per_cell; ++i)
        {
          auto face = cell->face(i);
          if (face->at_boundary())
          {
            // We get a lot of entry twice so we will need to get remove them
            for (unsigned int j=0; j<2; ++j)
            {
              Point<dim> vertex = face->vertex(j);
              types::global_dof_index dof_index = face->vertex_dof_index(j,0);
              interior_boundary.push_back(std::make_pair(vertex, dof_index));
            }
          }
        }
      }
    }
  }
  std::sort(interior_boundary.begin(), interior_boundary.end(), [](auto i, auto j)
            {
              return i.second < j.second;
            });
  auto end = std::unique(interior_boundary.begin(), interior_boundary.end());
  interior_boundary.resize(std::distance(interior_boundary.begin(), end));
}



template <int dim>
void HeatEquation<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << std::endl
            << "==========================================="
            << std::endl
            << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ true);
  sparsity_pattern.copy_from(dsp);

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  MatrixCreator::create_mass_matrix(dof_handler,
                                    QGauss<dim>(fe.degree+1),
                                    mass_matrix);
  Coefficient<dim> coef;
  MatrixCreator::create_laplace_matrix(dof_handler,
                                       QGauss<dim>(fe.degree+1),
                                       laplace_matrix,
                                       &coef);

  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void HeatEquation<dim>::solve_time_step()
{
  SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<> cg(solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.0);

  cg.solve(system_matrix, solution, system_rhs,
           preconditioner);

  constraints.distribute(solution);

  std::cout << "     " << solver_control.last_step()
            << " CG iterations." << std::endl;
}



template <int dim>
void HeatEquation<dim>::extract_values()
{
  unsigned int const n_values = interior_values.size();
  interior_values.resize(n_values);
  for (unsigned int i=0; i<n_values; ++i)
    interior_values[i] = solution[interior_boundary[i].second];
}



template <int dim>
void HeatEquation<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "U");

  data_out.build_patches();

  const std::string filename = "solution-"
                               + Utilities::int_to_string(timestep_number, 3) +
                               ".vtk";
  std::ofstream output(filename.c_str());
  data_out.write_vtk(output);
}


template <int dim>
void HeatEquation<dim>::run()
{
  int rc;

  build_mesh();

  setup_system();

  extract_boundary();

  Vector<double> tmp;
  Vector<double> forcing_terms;

  tmp.reinit (solution.size());
  forcing_terms.reinit (solution.size());


  VectorTools::interpolate(dof_handler,
                           ZeroFunction<dim>(),
                           old_solution);
  solution = old_solution;

  output_results();

  while (time <= 10)
    {
      time += time_step;
      ++timestep_number;

      std::cout << "Time step " << timestep_number << " at t=" << time
                << std::endl;

      mass_matrix.vmult(system_rhs, old_solution);

      laplace_matrix.vmult(tmp, old_solution);
      system_rhs.add(-(1 - theta) * time_step, tmp);

      RightHandSide<dim> rhs_function;
      rhs_function.set_time(time);
      VectorTools::create_right_hand_side(dof_handler,
                                          QGauss<dim>(fe.degree+1),
                                          rhs_function,
                                          tmp);
      forcing_terms = tmp;
      forcing_terms *= time_step * theta;

      rhs_function.set_time(time - time_step);
      VectorTools::create_right_hand_side(dof_handler,
                                          QGauss<dim>(fe.degree+1),
                                          rhs_function,
                                          tmp);

      forcing_terms.add(time_step * (1 - theta), tmp);

      system_rhs += forcing_terms;

      system_matrix.copy_from(mass_matrix);
      system_matrix.add(theta * time_step, laplace_matrix);

      constraints.condense (system_matrix, system_rhs);

      solve_time_step();
 
      extract_values();

      if (timestep_number % 10 == 0)
        output_results();

      old_solution = solution;

      /* Computation is completed, we let the driver know that the
         mesh is ready to be transfered to appA */
      rc = _app_notify_driver (hcq_from_driver);
      if (rc == -1)
      {
        fprintf (stderr, "ERROR: _app_notify_driver() failed\n");
      }

      /* We wait a signal from the driver to let us know that the
         mesh can be used again for another iteration. */
      rc = _app_wait (hcq_to_driver);
      if (rc == -1)
      {
        fprintf (stderr, "ERROR: _app_wait() failed\n");
      }

    }
}


int main()
{
  int rc;

  rc = _setup_hobbes (&db, NULL, NULL);
  if (rc != 0)
  {
    fprintf (stderr, "ERROR: _setup_hobbes() failed\n");
    goto exit_fn_on_error;
  }

  /*
   * Setting up 2 comamnd queues for bi-directional commands with the driver.
   */
  rc = _app_handshake (app, &hcq_to_driver, &hcq_from_driver);
  if (rc == -1)
  {
    fprintf (stderr, "ERROR: _app_handshake() failed\n");
    goto exit_fn_on_error;
  }

  // Notifying driver that we are up...
  rc = _app_notify_driver (hcq_from_driver);
  if (rc == -1)
  {
    fprintf (stderr, "ERROR: _app_notify_driver() failed\n");
    goto exit_fn_on_error;
  }

  // Waiting for the ACK from the driver before we start running...
  rc = _app_wait (hcq_to_driver);
  if (rc == -1)
  {
    fprintf (stderr, "ERROR: _app_wait() failed\n");
    goto exit_fn_on_error;
  }

  try
    {
      using namespace dealii;

      HeatEquation<2> heat_equation_solver;
      heat_equation_solver.run();

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      goto exit_fn_on_error;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
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
