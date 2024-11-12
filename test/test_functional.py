import unittest
from test.TestUtils import TestUtils
from sales import (predicted_sales_for_input, average_sales_for_region, total_sales_for_category,
                  highest_sales_region, trend_for_category, region_sales_distribution,
                  product_sales, predicted_sales_for_city, total_sales_distribution,
                   region_sales_comparison_difference, highest_sales_product, X, y, X_test, data)

class FunctionalTest(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()

        # Sample input and expected results
        self.example_input = X_test.iloc[0].tolist()
        self.expected_results = {
            "predicted_sales": 119.02169036865234,
            "average_sales_region_1": 240.40169694793536,
            "total_sales_category_2": 827455.8729999999,
            "highest_sales_region": 2,
            "trend_category_2": 456.4014743519029,
            "sales_distribution_region": {2: 389151.45900000003, 3: 710219.6845, 0: 492646.91320000007, 1: 669518.726},
            "total_average_sales_product_0": {'total': 25.227999999999998, 'average': 8.409333333333333},
            "predicted_sales_city_0": 25.5,
            "total_sales_distribution_category": {0: 728658.5757, 1: 705422.334, 2: 827455.8729999999},

            "difference_sales_region_1_2": -3.122370010763035,
            "highest_sales_product": 404
        }

    def test_predicted_sales_for_input(self):
        try:
            result = predicted_sales_for_input(self.example_input)
            if abs(result - self.expected_results["predicted_sales"]) < 0.1:
                self.test_obj.yakshaAssert("TestPredictedSalesForInput", True, "functional")
                print("TestPredictedSalesForInput = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictedSalesForInput", False, "functional")
                print("TestPredictedSalesForInput = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestPredictedSalesForInput", False, "functional")
            print("TestPredictedSalesForInput = Failed")

    def test_average_sales_for_region(self):
        try:
            result = average_sales_for_region(1, X, y)
            if abs(result - self.expected_results["average_sales_region_1"]) < 0.1:
                self.test_obj.yakshaAssert("TestAverageSalesForRegion", True, "functional")
                print("TestAverageSalesForRegion = Passed")
            else:
                self.test_obj.yakshaAssert("TestAverageSalesForRegion", False, "functional")
                print("TestAverageSalesForRegion = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestAverageSalesForRegion", False, "functional")
            print("TestAverageSalesForRegion = Failed")

    def test_total_sales_for_category(self):
        try:
            result = total_sales_for_category(2, X, y)
            if abs(result - self.expected_results["total_sales_category_2"]) < 0.1:
                self.test_obj.yakshaAssert("TestTotalSalesForCategory", True, "functional")
                print("TestTotalSalesForCategory = Passed")
            else:
                self.test_obj.yakshaAssert("TestTotalSalesForCategory", False, "functional")
                print("TestTotalSalesForCategory = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestTotalSalesForCategory", False, "functional")
            print("TestTotalSalesForCategory = Failed")

    def test_highest_sales_region(self):
        try:
            result = highest_sales_region(X, y)
            if result == self.expected_results["highest_sales_region"]:
                self.test_obj.yakshaAssert("TestHighestSalesRegion", True, "functional")
                print("TestHighestSalesRegion = Passed")
            else:
                self.test_obj.yakshaAssert("TestHighestSalesRegion", False, "functional")
                print("TestHighestSalesRegion = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestHighestSalesRegion", False, "functional")
            print("TestHighestSalesRegion = Failed")

    def test_trend_for_category(self):
        try:
            result = trend_for_category(2, X, y)
            if abs(result - self.expected_results["trend_category_2"]) < 0.1:
                self.test_obj.yakshaAssert("TestTrendForCategory", True, "functional")
                print("TestTrendForCategory = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrendForCategory", False, "functional")
                print("TestTrendForCategory = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestTrendForCategory", False, "functional")
            print("TestTrendForCategory = Failed")

    def test_region_sales_distribution(self):
        try:
            result = region_sales_distribution(X, y)
            if result == self.expected_results["sales_distribution_region"]:
                self.test_obj.yakshaAssert("TestRegionSalesDistribution", True, "functional")
                print("TestRegionSalesDistribution = Passed")
            else:
                self.test_obj.yakshaAssert("TestRegionSalesDistribution", False, "functional")
                print("TestRegionSalesDistribution = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestRegionSalesDistribution", False, "functional")
            print("TestRegionSalesDistribution = Failed")

    def test_product_sales(self):
        try:
            result = product_sales(0, data, y)
            if result == self.expected_results["total_average_sales_product_0"]:
                self.test_obj.yakshaAssert("TestProductSales", True, "functional")
                print("TestProductSales = Passed")
            else:
                self.test_obj.yakshaAssert("TestProductSales", False, "functional")
                print("TestProductSales = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestProductSales", False, "functional")
            print("TestProductSales = Failed")

    def test_predicted_sales_for_city(self):
        try:
            result = predicted_sales_for_city(0, data, y)
            if abs(result - self.expected_results["predicted_sales_city_0"]) < 0.1:
                self.test_obj.yakshaAssert("TestPredictedSalesForCity", True, "functional")
                print("TestPredictedSalesForCity = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictedSalesForCity", False, "functional")
                print("TestPredictedSalesForCity = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestPredictedSalesForCity", False, "functional")
            print("TestPredictedSalesForCity = Failed")

    def test_total_sales_distribution(self):
        try:
            result = total_sales_distribution(X, y)
            if result == self.expected_results["total_sales_distribution_category"]:
                self.test_obj.yakshaAssert("TestTotalSalesDistribution", True, "functional")
                print("TestTotalSalesDistribution = Passed")
            else:
                self.test_obj.yakshaAssert("TestTotalSalesDistribution", False, "functional")
                print("TestTotalSalesDistribution = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestTotalSalesDistribution", False, "functional")
            print("TestTotalSalesDistribution = Failed")



    def test_region_sales_comparison_difference(self):
        try:
            result = region_sales_comparison_difference(1, 2, X, y)
            if abs(result - self.expected_results["difference_sales_region_1_2"]) < 0.1:
                self.test_obj.yakshaAssert("TestRegionSalesComparisonDifference", True, "functional")
                print("TestRegionSalesComparisonDifference = Passed")
            else:
                self.test_obj.yakshaAssert("TestRegionSalesComparisonDifference", False, "functional")
                print("TestRegionSalesComparisonDifference = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestRegionSalesComparisonDifference", False, "functional")
            print("TestRegionSalesComparisonDifference = Failed")

    def test_highest_sales_product(self):
        try:
            result = highest_sales_product(data, y)
            if result == self.expected_results["highest_sales_product"]:
                self.test_obj.yakshaAssert("TestHighestSalesProduct", True, "functional")
                print("TestHighestSalesProduct = Passed")
            else:
                self.test_obj.yakshaAssert("TestHighestSalesProduct", False, "functional")
                print("TestHighestSalesProduct = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestHighestSalesProduct", False, "functional")
            print("TestHighestSalesProduct = Failed")

