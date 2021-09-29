import unittest

from dimod import sym, BINARY, INTEGER

from job_shop_scheduler import JSSCQM
from data import Data
import utils.plot_schedule as job_plotter


class TestData(unittest.TestCase):
    def test_data(self):
        """Test input data name, size and max completion time"""

        input_file = "tests/instance_test.txt"
        test_data = Data(input_file)
        self.assertEqual(test_data.instance_name, 'instance_test')
        test_data.read_input_data()
        self.assertEqual(test_data.num_jobs, 3)
        self.assertEqual(test_data.num_machines, 3)
        self.assertEqual(test_data.max_makespan, 14)

    def test_prep_solution_for_plotting(self):
        """Test if data is formatted correctly for plotting"""

        input_file = "tests/instance_test.txt"
        test_data = Data(input_file)
        test_data.read_input_data()
        solution = {(0, 0): 3.0, (1, 0): 5.0, (2, 0): 4.0, (0, 1): 5.0,
                    (1, 1): 1.0, (2, 1): 2.0, (0, 2): 1.0, (1, 2): 3.0,
                    (2, 2): 0.0}

        job_start_time, processing_time = \
            job_plotter.plot_solution(test_data, solution)
        self.assertEqual({0: [3.0, 5.0, 1.0],
                          1: [5.0, 1.0, 3.0],
                          2: [4.0, 2.0, 0.0]}, job_start_time)

        self.assertEqual({0: [1, 2, 2], 1: [2, 1, 2], 2: [1, 2, 1]},
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
        self.assertEqual(num_binaries, 9)
        num_integers = sum(cqm.vartype(v) is INTEGER for v in cqm.variables)
        self.assertEqual(num_integers, 10)
        num_linear_constraints = sum(
            constraint.lhs.is_linear() for constraint in
            cqm.constraints.values())
        self.assertEqual(num_linear_constraints, 9)
        num_quadratic_constraints = sum(
            not constraint.lhs.is_linear() for constraint in
            cqm.constraints.values())
        self.assertEqual(num_quadratic_constraints, 9)
        num_ge_inequality_constraints = sum(
            constraint.sense is sym.Sense.Ge for constraint in
            cqm.constraints.values())
        self.assertEqual(num_ge_inequality_constraints, 18)


if __name__ == '__main__':
    unittest.main()
