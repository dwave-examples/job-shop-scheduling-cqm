import os
import subprocess
import sys
import unittest

from dimod import BINARY, INTEGER, sym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../')
import src.utils.plot_schedule as job_plotter
from src.job_shop_scheduler import JobShopSchedulingCQM
from src.model_data import JobShopData

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestSmoke(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_smoke(self):
        """Run job_shop_scheduler.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'src', 'job_shop_scheduler.py')
        subprocess.check_output([sys.executable, demo_file])


class TestData(unittest.TestCase):
    def test_data(self):
        """Test input data name, size and max completion time"""

        input_file = "tests/instance_test.txt"
        test_model_data = JobShopData()
        test_model_data.load_from_file(input_file)
        
        self.assertEqual(len(test_model_data.jobs), 3)
        self.assertEqual(len(test_model_data.resources), 3)
        self.assertEqual(test_model_data.get_max_makespan(), 24)


    def test_prep_solution_for_plotting(self):
        """Test if data is formatted correctly for plotting"""

        input_file = "tests/instance_test.txt"
        test_data = JobShopData()
        test_data.load_from_file(input_file)

        solution = {('0', 0): (1, 8.0, 2), ('1', 0): (1, 11.0, 3),
                    ('2', 0): (1, 14.0, 0), ('0', 1): (2, 0.0, 3),
                    ('1', 1): (2, 3.0, 2), ('2', 1): (2, 5.0, 3),
                    ('0', 2): (0, 3.0, 4), ('1', 2): (0, 7.0, 2),
                    ('2', 2): (0, 9.0, 5)}

        job_start_time, processing_time = \
            job_plotter.plot_solution(test_data, solution)
        self.assertEqual({'0': [8.0, 0.0, 3.0],
                          '1': [11.0, 3.0, 7.0],
                          '2': [14.0, 5.0, 9.0]}, job_start_time)

        self.assertEqual({'0': [2, 3, 4], '1': [3, 2, 2], '2': [0, 3, 5]},
                         processing_time)


    def test_jss_cqm_size(self):
        "Testing size of CQM model built for an JSS instance"""

        input_file = "tests/instance_test.txt"
        model_data = JobShopData()
        model_data.load_from_file(input_file)
        model = JobShopSchedulingCQM(model_data)
        model.define_cqm_model()
        model.define_variables(model_data)
        model.add_precedence_constraints(model_data)
        model.add_quadratic_overlap_constraint(model_data)
        model.add_makespan_constraint(model_data)
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


if __name__ == '__main__':
    unittest.main()