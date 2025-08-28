#define _GLIBCXX_USE_CXX11_ABI 0 // Align the ABI version to avoid compatibility issues with `Placment.h`
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "ObjectiveFunction.h"
#include "Point.h"

/**
 * @brief Base class for optimizers
 */
class BaseOptimizer
{
public:
    /////////////////////////////////
    // Constructors
    /////////////////////////////////

    BaseOptimizer(BaseFunction &obj, std::vector<Point2<double>> &var)
        : obj_(obj), var_(var) {}

    /////////////////////////////////
    // Methods
    /////////////////////////////////

    // Initialize the optimizer, e.g., do some calculation for the first step.
    virtual void Initialize() = 0;

    // Perform one optimization step
    // For example, update var_ based on the gradient of obj_:
    //     var_ = var_ - learning_rate * obj_.grad();
    virtual void Step() = 0;

protected:
    /////////////////////////////////
    // Data members
    /////////////////////////////////

    BaseFunction &obj_;                // Objective function to optimize
    std::vector<Point2<double>> &var_; // Variables to optimize
};

/**
 * @brief Simple conjugate gradient optimizer with constant step size
 */
class SimpleConjugateGradient : public BaseOptimizer
{
public:
    /////////////////////////////////
    // Constructors
    /////////////////////////////////

    SimpleConjugateGradient(BaseFunction &obj, std::vector<Point2<double>> &var, const double &grid_num, Placement &placement);

    /////////////////////////////////
    // Methods
    /////////////////////////////////

    // Set the scaling factor for the step size
    inline void setScalingFactor(double scaling_factor) { scaling_factor_ = scaling_factor; }
    inline double getScalingFactor() const { return scaling_factor_; }

    // Initialize the optimizer
    void Initialize() override;

    // Perform one optimization step
    void Step() override;
    inline double getAlpha() const { return alpha_; }

private:
    /////////////////////////////////
    // Data members
    /////////////////////////////////

    std::vector<Point2<double>> grad_prev_; // Gradient of the objective function at the previous
                                            // step, i.e., g_{k-1} in the NTUPlace3 paper
    std::vector<Point2<double>> dir_prev_;  // Direction of the previous step,
                                            // i.e., d_{k-1} in the NTUPlace3 paper
    size_t step_;                           // Current step number
    double alpha_;                          // Step size
    double w_b_;                            // Width of the bin
    double h_b_;                            // Height of the bin
    double grid_num_;                       // Number of bins in one dimension
    double scaling_factor_;                 // Scaling factor for the step size
    Placement &placement_;
};

// TODO(Optional): Implement your own optimizer.

#endif // OPTIMIZER_H
