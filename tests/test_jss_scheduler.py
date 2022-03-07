import unittest
import os
import sys
import subprocess

from dimod import sym, BINARY, INTEGER, ConstrainedQuadraticModel

from job_shop_scheduler import JSSCQM
from data import Data
import utils.plot_schedule as job_plotter

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestSmoke(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_smoke(self):
        """Run job_shop_scheduler.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'job_shop_scheduler.py')
        subprocess.check_output([sys.executable, demo_file])


class TestData(unittest.TestCase):
    def test_data(self):
        """Test input data name, size and max completion time"""

        input_file = "tests/instance_test.txt"
        test_data = Data(input_file)
        self.assertEqual(test_data.instance_name, 'instance_test')
        test_data.read_input_data()
        self.assertEqual(test_data.num_jobs, 3)
        self.assertEqual(test_data.num_machines, 3)
        self.assertEqual(test_data.max_makespan, 24)

    def test_prep_solution_for_plotting(self):
        """Test if data is formatted correctly for plotting"""

        input_file = "tests/instance_test.txt"
        test_data = Data(input_file)
        test_data.read_input_data()
        solution = {(0, 0): (1, 8.0, 2), (1, 0): (1, 11.0, 3),
                    (2, 0): (1, 14.0, 0), (0, 1): (2, 0.0, 3),
                    (1, 1): (2, 3.0, 2), (2, 1): (2, 5.0, 3),
                    (0, 2): (0, 3.0, 4), (1, 2): (0, 7.0, 2),
                    (2, 2): (0, 9.0, 5)}

        job_start_time, processing_time = \
            job_plotter.plot_solution(test_data, solution)

        self.assertEqual({0: [8.0, 0.0, 3.0],
                          1: [11.0, 3.0, 7.0],
                          2: [14.0, 5.0, 9.0]}, job_start_time)

        self.assertEqual({0: [2, 3, 4], 1: [3, 2, 2], 2: [0, 3, 5]},
                         processing_time)

    def test_jss_cqm_size(self):
        "Testing size of CQM model built for an JSS instance"""

        input_file = "tests/instance_test.txt"
        test_data = Data(input_file)
        test_data.read_input_data()
        model = JSSCQM()
        model.define_cqm_model()
        model.define_variables(test_data)
        model.add_precedence_constraints(test_data)
        model.add_quadratic_overlap_constraint(test_data)
        model.add_makespan_constraint(test_data)
        model.define_objective_function()
        cqm = model.cqm
        num_binaries = sum(cqm.vartype(v) is BINARY for v in cqm.variables)
        self.assertEqual(num_binaries, 7)
        num_integers = sum(cqm.vartype(v) is INTEGER for v in cqm.variables)
        self.assertEqual(num_integers, 10)
        num_linear_constraints = sum(
            constraint.lhs.is_linear() for constraint in
            cqm.constraints.values())
        self.assertEqual(num_linear_constraints, 9)
        num_quadratic_constraints = sum(
            not constraint.lhs.is_linear() for constraint in
            cqm.constraints.values())
        self.assertEqual(num_quadratic_constraints, 7)
        num_ge_inequality_constraints = sum(
            constraint.sense is sym.Sense.Ge for constraint in
            cqm.constraints.values())
        self.assertEqual(num_ge_inequality_constraints, 16)


class CQM_model(unittest.TestCase):
    def test_model(self):
        """Test if the cqm gives correct energy for a given sample"""

        # Check energy for an infeasible sample
        sample = {'x0_2': 3.0, 'x0_1': 0.0, 'x0_0': 10.0, 'x1_2': 10.0,
                  'x1_1': 6.0, 'x1_0': 14.0, 'x2_2': 14.0, 'x2_1': 8.0,
                  'x2_0': 20.0, 'y0_1_0': 1.0, 'y0_1_1': 1.0, 'y0_1_2': 1.0,
                  'y0_2_0': 1.0, 'y0_2_1': 1.0, 'y0_2_2': 1.0, 'y1_2_0': 1.0,
                  'y1_2_1': 1.0, 'y1_2_2': 1.0, 'makespan': 20.0}

        expected_violations = {
            'pj0_m1': -0.0, 'pj0_m2': -3.0, 'pj1_m1': -2.0,
            'pj1_m2': -2.0,
            'pj2_m1': -3.0, 'pj2_m2': -1.0,
            'OneJobj0_j1_m0': -2.0,
            'OneJobj0_j1_m1': -3.0, 'OneJobj0_j1_m2': -3.0,
            'OneJobj0_j2_m1': -5.0, 'OneJobj0_j2_m2': -7.0,
            'OneJobj1_j2_m1': -0.0, 'OneJobj1_j2_m2': -2.0,
            'makespan_ctr0': -8.0, 'makespan_ctr1': -3.0,
            'makespan_ctr2': -0.0
        }

        input_file = "tests/instance_test.txt"
        test_data = Data(input_file)
        test_data.read_input_data()
        model = JSSCQM()
        model.define_cqm_model()
        model.define_variables(test_data)
        model.add_precedence_constraints(test_data)
        model.add_quadratic_overlap_constraint(test_data)
        model.add_makespan_constraint(test_data)
        model.define_objective_function()
        violations = {label: violation for (label, violation)
                      in ConstrainedQuadraticModel.iter_violations(model.cqm,
                                                                   sample)}
        self.assertEqual(violations, expected_violations)
        self.assertTrue(model.cqm.check_feasible(sample))

        # Check energy for an infeasible sample
        infeasible_sample = sample.copy()
        infeasible_sample['x0_2'] = 7.0
        infeasible_sample['make_span'] = 16.0
        violations = {label: violation for (label, violation)
                      in ConstrainedQuadraticModel.iter_violations(
                model.cqm, infeasible_sample)}
        expected_violations = {
            'pj0_m1': -4.0, 'pj0_m2': 1.0, 'pj1_m1': -2.0, 'pj1_m2': -2.0,
            'pj2_m1': -3.0, 'pj2_m2': -1.0, 'OneJobj0_j1_m0': -2.0,
            'OneJobj0_j1_m1': -3.0, 'OneJobj0_j1_m2': 1.0,
            'OneJobj0_j2_m1': -5.0, 'OneJobj0_j2_m2': -3.0,
            'OneJobj1_j2_m1': -0.0, 'OneJobj1_j2_m2': -2.0,
            'makespan_ctr0': -8.0, 'makespan_ctr1': -3.0,
            'makespan_ctr2': -0.0}
        self.assertEqual(violations, expected_violations)
        self.assertFalse(model.cqm.check_feasible(infeasible_sample))
