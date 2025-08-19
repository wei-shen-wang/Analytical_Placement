#define _GLIBCXX_USE_CXX11_ABI 0 // Align the ABI version to avoid compatibility issues with `Placment.h`
#ifndef OBJECTIVEFUNCTION_H
#define OBJECTIVEFUNCTION_H

#include <vector>
#include <set>
#include <map>

#include "Placement.h"
#include "Point.h"

/**
 * @brief Base class for objective functions
 */
class BaseFunction
{
public:
    /////////////////////////////////
    // Conssutructors
    /////////////////////////////////

    BaseFunction(const size_t &input_size) : grad_(input_size) {}

    /////////////////////////////////
    // Accessors
    /////////////////////////////////

    const std::vector<Point2<double>> &grad() const { return grad_; }
    const double &value() const { return value_; }
    const double getSumOfNormPartialDerivative();


    /////////////////////////////////
    // Methods
    /////////////////////////////////

    // Forward pass, compute the value of the function
    virtual const double &operator()(const std::vector<Point2<double>> &input) = 0;

    // Backward pass, compute the gradient of the function
    virtual const std::vector<Point2<double>> &Backward() = 0;

protected:
    /////////////////////////////////
    // Data members
    /////////////////////////////////

    std::vector<Point2<double>> grad_; // Gradient of the function
    double value_;                     // Value of the function
};

/**
 * @brief Wirelength function
 */
class Wirelength : public BaseFunction
{
    // TODO: Implement the wirelength function, add necessary data members for caching
public:
    Wirelength(Placement &placement);
    /////////////////////////////////
    // Methods
    /////////////////////////////////

    const double &operator()(const std::vector<Point2<double>> &input) override;
    const std::vector<Point2<double>> &Backward() override;
private:
    Placement &placement_; // Reference to the placement object
    std::vector<Point2<double>> input_; // Cache the input for backward pass
    double gamma_;
};

/**
 * @brief Density function
 */
class Density : public BaseFunction
{
    // TODO: Implement the density function, add necessary data members for caching
public:
    Density(Placement &placement, const double &grid_num);
    /////////////////////////////////
    // Methods
    /////////////////////////////////

    const double &operator()(const std::vector<Point2<double>> &input) override;
    const std::vector<Point2<double>> &Backward() override;
    double calculateOverflowRatio();

private:
    /////////////////////////////////
    // Data members
    /////////////////////////////////

    std::vector<Point2<double>> input_; // Cache the input for backward pass
    Placement &placement_;
    int grid_num_;
    std::vector<double> bin_density_minus_movable_area_;
    std::vector<double> init_bin_density_minus_movable_area_;
    std::vector<std::set<int>> module_to_bin_indexes_;
    double w_b_;
    double h_b_;
};

/**
 * @brief Objective function for global placement
 */
class ObjectiveFunction : public BaseFunction
{
    // TODO: Implement the objective function for global placement, add necessary data
    // members for caching
    //
    // Hint: The objetive function of global placement is as follows:
    //       f(t) = wirelength(t) + lambda * density(t),
    // where t is the positions of the modules, and lambda is the penalty weight.
    // You may need an interface to update the penalty weight (lambda) dynamically.
public:
    ObjectiveFunction(Placement &placement, double grid_num, Wirelength &wirelength, Density &density);
    /////////////////////////////////
    // Methods
    /////////////////////////////////

    const double &operator()(const std::vector<Point2<double>> &input) override;
    const std::vector<Point2<double>> &Backward() override;
    void setLambda(double lambda) {  lambda_ = lambda; };

private:
    Placement &placement_; // Reference to the placement object
    double grid_num_; // Number of bins in one dimension
    Wirelength &wirelength_;
    Density &density_;
    double lambda_; // Penalty weight for density
};

#endif // OBJECTIVEFUNCTION_H
