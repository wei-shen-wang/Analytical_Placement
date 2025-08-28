#include "ObjectiveFunction.h"

#include "cstdio"
#include <omp.h>
#include <limits>
#include <cassert>

const double BaseFunction::getSumOfNormPartialDerivative()
{
    double sum = 0.0;
    for (const Point2<double> &grad : grad_)
    {
        // sum += std::sqrt(grad.x * grad.x + grad.y * grad.y);
        sum += std::abs(grad.x) + std::abs(grad.y);
    }
    return sum;
}

Wirelength::Wirelength(Placement &placement)
    : BaseFunction(placement.numModules()), placement_(placement)
{
    gamma_ = 0.01 * placement_.rectangleChip().width();
}

const double &Wirelength::operator()(const std::vector<Point2<double>> &input)
{
    value_ = 0.;
    input_ = input; // Cache the input for backward pass
    grad_.assign(placement_.numModules(), Point2<double>(0.0, 0.0));

    // LSE
    // for (size_t i = 0;i < placement_.numNets(); ++i)
    // {
    //     Net &net = placement_.net(i);
    //     double sum_exp_x = 0.0;
    //     double sum_exp_neg_x = 0.0;
    //     double sum_exp_y = 0.0;
    //     double sum_exp_neg_y = 0.0;
    //     for (size_t j = 0; j < net.numPins(); ++j)
    //     {
    //         Pin &pin = net.pin(j);
    //         unsigned module_id = pin.moduleId();
    //         double pin_x_pos = input[module_id].x + pin.xOffset();
    //         double pin_y_pos = input[module_id].y + pin.yOffset();
    //         double exp_x = exp(pin_x_pos / gamma_);
    //         double exp_neg_x = 1.0 / exp_x;
    //         double exp_y = exp(pin_y_pos / gamma_);
    //         double exp_neg_y = 1.0 / exp_y;
    //         sum_exp_x += exp_x;
    //         sum_exp_neg_x += exp_neg_x;
    //         sum_exp_y += exp_y;
    //         sum_exp_neg_y += exp_neg_y;
    //     }
    //     value_ += log(sum_exp_x) + log(sum_exp_neg_x) + log(sum_exp_y) + log(sum_exp_neg_y);
    //     // Compute the gradient for each x and y
    //     for (size_t j = 0; j < net.numPins(); ++j)
    //     {
    //         Pin &pin = net.pin(j);
    //         unsigned module_id = pin.moduleId();
    //         if (placement_.module(module_id).isFixed()) {
    //             continue; // Skip fixed modules
    //         }
    //         double pin_x_pos = input[module_id].x + pin.xOffset();
    //         double pin_y_pos = input[module_id].y + pin.yOffset();
    //         double exp_x = exp(pin_x_pos / gamma_);
    //         double exp_neg_x = 1.0 / exp_x;
    //         double exp_y = exp(pin_y_pos / gamma_);
    //         double exp_neg_y = 1.0 / exp_y;
    //         double grad_x = gamma_ * (exp_x / sum_exp_x - exp_neg_x / sum_exp_neg_x);
    //         double grad_y = gamma_ * (exp_y / sum_exp_y - exp_neg_y / sum_exp_neg_y);
    //         // Update the gradient for the module
    //         grad_[module_id].x += grad_x;
    //         grad_[module_id].y += grad_y;
    //     }
    // }
    // value_ *= gamma_;

    // normal WA
    #pragma omp parallel for
    for (size_t i = 0; i < placement_.numNets(); ++i)
    {
        Net &net = placement_.net(i);
        double WA_x_max_numerator = 0.0, WA_x_min_numerator = 0.0, WA_y_max_numerator = 0.0, WA_y_min_numerator = 0.0;
        double WA_x_max_denominator = 0.0, WA_x_min_denominator = 0.0, WA_y_max_denominator = 0.0, WA_y_min_denominator = 0.0;
        for (size_t j = 0; j < net.numPins(); ++j)
        {
            Pin &pin = net.pin(j);
            unsigned module_id = pin.moduleId();
            double pin_x_pos = input[module_id].x + pin.xOffset();
            double pin_y_pos = input[module_id].y + pin.yOffset();
            double exp_x_max = exp(pin_x_pos / gamma_);
            double exp_x_min = 1.0 / exp_x_max;
            double exp_y_max = exp(pin_y_pos / gamma_);
            double exp_y_min = 1.0 / exp_y_max;
            WA_x_max_numerator += pin_x_pos * exp_x_max;
            WA_x_min_numerator += pin_x_pos * exp_x_min;
            WA_y_max_numerator += pin_y_pos * exp_y_max;
            WA_y_min_numerator += pin_y_pos * exp_y_min;
            WA_x_max_denominator += exp_x_max;
            WA_x_min_denominator += exp_x_min;
            WA_y_max_denominator += exp_y_max;
            WA_y_min_denominator += exp_y_min;
        }
        double WA_x_max = WA_x_max_numerator / WA_x_max_denominator;
        double WA_x_min = WA_x_min_numerator / WA_x_min_denominator;
        double WA_y_max = WA_y_max_numerator / WA_y_max_denominator;
        double WA_y_min = WA_y_min_numerator / WA_y_min_denominator;
        double WA_wirelength = (WA_x_max - WA_x_min) + (WA_y_max - WA_y_min);

        
        // Compute the gradient for each x and y
        for (size_t j = 0; j < net.numPins(); ++j)
        {
            Pin &pin = net.pin(j);
            unsigned module_id = pin.moduleId();
            if (placement_.module(module_id).isFixed())
            {
                continue; // Skip fixed modules
            }
            double pin_x_pos = input[module_id].x + pin.xOffset();
            double pin_y_pos = input[module_id].y + pin.yOffset();
            double exp_x_max = exp(pin_x_pos / gamma_);
            double exp_x_min = 1.0 / exp_x_max;
            double exp_y_max = exp(pin_y_pos / gamma_);
            double exp_y_min = 1.0 / exp_y_max;
            double WA_x_max_nominator_grad = (1 + pin_x_pos / gamma_) * exp_x_max;
            double WA_x_min_nominator_grad = (1 - pin_x_pos / gamma_) * exp_x_min;
            double WA_y_max_nominator_grad = (1 + pin_y_pos / gamma_) * exp_y_max;
            double WA_y_min_nominator_grad = (1 - pin_y_pos / gamma_) * exp_y_min;
            double WA_x_max_denominator_grad = exp_x_max / gamma_;
            double WA_x_min_denominator_grad = -exp_x_min / gamma_;
            double WA_y_max_denominator_grad = exp_y_max / gamma_;
            double WA_y_min_denominator_grad = -exp_y_min / gamma_;
            double WA_x_max_grad = (WA_x_max_nominator_grad * WA_x_max_denominator - WA_x_max_numerator * WA_x_max_denominator_grad) / pow(WA_x_max_denominator, 2);
            double WA_x_min_grad = (WA_x_min_nominator_grad * WA_x_min_denominator - WA_x_min_numerator * WA_x_min_denominator_grad) / pow(WA_x_min_denominator, 2);
            double WA_y_max_grad = (WA_y_max_nominator_grad * WA_y_max_denominator - WA_y_max_numerator * WA_y_max_denominator_grad) / pow(WA_y_max_denominator, 2);
            double WA_y_min_grad = (WA_y_min_nominator_grad * WA_y_min_denominator - WA_y_min_numerator * WA_y_min_denominator_grad) / pow(WA_y_min_denominator, 2);
            double grad_x = WA_x_max_grad - WA_x_min_grad;
            double grad_y = WA_y_max_grad - WA_y_min_grad;
            
            // Update the gradient for the module
            #pragma omp atomic
            grad_[module_id].x += grad_x;
            #pragma omp atomic
            grad_[module_id].y += grad_y;
        }
        #pragma omp atomic
        value_ += WA_wirelength;
    }

    // stable WA
    // for (size_t i = 0; i < placement_.numNets(); ++i)
    // {
    //     Net &net = placement_.net(i);
    //     double x_max = 0., x_min = 0., y_max = 0., y_min = 0.;
    //     for (size_t j = 0; j < net.numPins(); ++j)
    //     {
    //         Pin &pin = net.pin(j);
    //         unsigned module_id = pin.moduleId();
    //         if (j == 0) {
    //             x_max = input[module_id].x + pin.xOffset();
    //             x_min = input[module_id].x + pin.xOffset();
    //             y_max = input[module_id].y + pin.yOffset();
    //             y_min = input[module_id].y + pin.yOffset();
    //         }
    //         x_max = std::max(x_max, input[module_id].x + pin.xOffset());
    //         x_min = std::min(x_min, input[module_id].x + pin.xOffset());
    //         y_max = std::max(y_max, input[module_id].y + pin.yOffset());
    //         y_min = std::min(y_min, input[module_id].y + pin.yOffset());
    //     }
    //     double WA_x_max_numerator = 0.0;
    //     double WA_x_min_numerator = 0.0;
    //     double WA_y_max_numerator = 0.0;
    //     double WA_y_min_numerator = 0.0;
    //     double WA_x_max_denominator = 0.0;
    //     double WA_x_min_denominator = 0.0;
    //     double WA_y_max_denominator = 0.0;
    //     double WA_y_min_denominator = 0.0;
    //     for (size_t j = 0; j < net.numPins(); ++j)
    //     {
    //         Pin &pin = net.pin(j);
    //         unsigned module_id = pin.moduleId();
    //         double pin_x_pos = input[module_id].x + pin.xOffset();
    //         double pin_y_pos = input[module_id].y + pin.yOffset();
    //         double exp_x_max = exp((pin_x_pos - x_max) / gamma_);
    //         double exp_x_min = exp((x_min - pin_x_pos) / gamma_);
    //         double exp_y_max = exp((pin_y_pos - y_max) / gamma_);
    //         double exp_y_min = exp((y_min - pin_y_pos) / gamma_);
    //         WA_x_max_numerator += pin_x_pos * exp_x_max;
    //         WA_x_min_numerator += pin_x_pos * exp_x_min;
    //         WA_y_max_numerator += pin_y_pos * exp_y_max;
    //         WA_y_min_numerator += pin_y_pos * exp_y_min;
    //         WA_x_max_denominator += exp_x_max;
    //         WA_x_min_denominator += exp_x_min;
    //         WA_y_max_denominator += exp_y_max;
    //         WA_y_min_denominator += exp_y_min;
    //     }
    //     double WA_x_max = WA_x_max_numerator / WA_x_max_denominator;
    //     double WA_x_min = WA_x_min_numerator / WA_x_min_denominator;
    //     double WA_y_max = WA_y_max_numerator / WA_y_max_denominator;
    //     double WA_y_min = WA_y_min_numerator / WA_y_min_denominator;
    //     double WA_wirelength = (WA_x_max - WA_x_min) + (WA_y_max - WA_y_min);
    //     value_ += WA_wirelength;
    //     // Compute the gradient for each x and y
    //     for (size_t j = 0; j < net.numPins(); ++j)
    //     {
    //         Pin &pin = net.pin(j);
    //         unsigned module_id = pin.moduleId();
    //         double pin_x_pos = input[module_id].x + pin.xOffset();
    //         double pin_y_pos = input[module_id].y + pin.yOffset();
    //         double exp_x_max = exp((pin_x_pos - x_max) / gamma_);
    //         double exp_x_min = exp((x_min - pin_x_pos) / gamma_);
    //         double exp_y_max = exp((pin_y_pos - y_max) / gamma_);
    //         double exp_y_min = exp((y_min - pin_y_pos) / gamma_);
    //         double WA_x_max_nominator_grad = (1 + pin_x_pos / gamma_) * exp_x_max;
    //         double WA_x_min_nominator_grad = (1 - pin_x_pos / gamma_) * exp_x_min;
    //         double WA_y_max_nominator_grad = (1 + pin_y_pos / gamma_) * exp_y_max;
    //         double WA_y_min_nominator_grad = (1 - pin_y_pos / gamma_) * exp_y_min;
    //         double WA_x_max_denominator_grad = exp_x_max / gamma_;
    //         double WA_x_min_denominator_grad = - exp_x_min / gamma_;
    //         double WA_y_max_denominator_grad = exp_y_max / gamma_;
    //         double WA_y_min_denominator_grad = - exp_y_min / gamma_;
    //         double WA_x_max_grad = (WA_x_max_nominator_grad * WA_x_max_denominator - WA_x_max_numerator * WA_x_max_denominator_grad) / pow(WA_x_max_denominator, 2);
    //         double WA_x_min_grad = (WA_x_min_nominator_grad * WA_x_min_denominator - WA_x_min_numerator * WA_x_min_denominator_grad) / pow(WA_x_min_denominator, 2);
    //         double WA_y_max_grad = (WA_y_max_nominator_grad * WA_y_max_denominator - WA_y_max_numerator * WA_y_max_denominator_grad) / pow(WA_y_max_denominator, 2);
    //         double WA_y_min_grad = (WA_y_min_nominator_grad * WA_y_min_denominator - WA_y_min_numerator * WA_y_min_denominator_grad) / pow(WA_y_min_denominator, 2);
    //         double grad_x = WA_x_max_grad - WA_x_min_grad;
    //         double grad_y = WA_y_max_grad - WA_y_min_grad;
    //         // Update the gradient for the module
    //         grad_[module_id].x += grad_x;
    //         grad_[module_id].y += grad_y;
    //     }
    // }
    return value_;
}

const std::vector<Point2<double>> &Wirelength::Backward()
{
    return grad_;
}

Density::Density(Placement &placement, const double &grid_num) : BaseFunction(placement.numModules()), placement_(placement), grid_num_(grid_num)
{
    w_b_ = (placement_.boundryRight() - placement_.boundryLeft()) / grid_num_;
    h_b_ = (placement_.boundryTop() - placement_.boundryBottom()) / grid_num_;
    grid_num_squared_ = grid_num_ * grid_num_;
    bin_density_minus_movable_area_ = std::vector<double>(grid_num_squared_, -w_b_ * h_b_);
    module_to_bin_indexes_ = std::vector<std::set<int>>(placement_.numModules());
    printf("bin width is %.4f, bin height is %.4f", w_b_, h_b_);
    printf("    Placement boundry: (%.f,%.f)-(%.f,%.f)\n", placement_.boundryLeft(), placement_.boundryBottom(),
           placement_.boundryRight(), placement_.boundryTop());
}

const double &Density::operator()(const std::vector<Point2<double>> &input)
{
    const size_t &kNumModule = placement_.numModules();
    std::fill(bin_density_minus_movable_area_.begin(), bin_density_minus_movable_area_.end(), -w_b_ * h_b_);
    // module_to_bin_indexes_ = std::vector<std::set<int>>(kNumModule);
    for (std::set<int> &s : module_to_bin_indexes_)
    {
        s.clear();
    }
    value_ = 0.;

    #pragma omp parallel for
    for (size_t k = 0; k < kNumModule; ++k)
    {
        Module &cur_module = placement_.module(k);
        if (cur_module.isFixed())
        {
            continue;
        }
        const double &w_v = cur_module.width();
        const double &h_v = cur_module.height();
        double left_most_bin_center_x = input[k].x - w_b_ * 2 - w_v / 2;
        double right_most_bin_center_x = input[k].x + w_b_ * 2 + w_v / 2;
        double bottom_most_bin_center_y = input[k].y - h_b_ * 2 - h_v / 2;
        double top_most_bin_center_y = input[k].y + h_b_ * 2 + h_v / 2;
        int left_bin_index = std::max(0, (int)std::floor((left_most_bin_center_x - placement_.boundryLeft()) / w_b_));
        int right_bin_index = std::min(grid_num_ - 1, (int)std::ceil((right_most_bin_center_x - placement_.boundryLeft()) / w_b_));
        int bottom_bin_index = std::max(0, (int)std::floor((bottom_most_bin_center_y - placement_.boundryBottom()) / h_b_));
        int top_bin_index = std::min(grid_num_ - 1, (int)std::ceil((top_most_bin_center_y - placement_.boundryBottom()) / h_b_));
        // traverse through the bins
        std::map<int, double> bin_index_to_pxpy;
        double cell_sum_pxpy = 0.;
        for (int i = left_bin_index; i <= right_bin_index; ++i)
        {
            for (int j = bottom_bin_index; j <= top_bin_index; ++j)
            {
                double bin_center_x = placement_.boundryLeft() + (i + 0.5) * w_b_;
                double bin_center_y = placement_.boundryBottom() + (j + 0.5) * h_b_;
                Rectangle bin_rect_4x = Rectangle(bin_center_x - w_b_ * 2, bin_center_y - h_b_ * 2, bin_center_x + w_b_ * 2, bin_center_y + h_b_ * 2);
                double overlap = Rectangle::overlapArea(Rectangle(input[k].x - w_v / 2, input[k].y - h_v / 2, input[k].x + w_v / 2, input[k].y + h_v / 2), bin_rect_4x);
                if (overlap == 0.)
                {
                    continue;
                }

                int bin_index = i * grid_num_ + j;

                module_to_bin_indexes_[k].emplace(bin_index);

                double p_x = 0., p_y = 0.;
                double d_x = std::abs(input[k].x - bin_center_x);
                double d_y = std::abs(input[k].y - bin_center_y);
                if (d_x <= (w_v / 2 + w_b_))
                {
                    double a_x = 4. / ((w_v + 2. * w_b_) * (w_v + 4. * w_b_));
                    p_x = 1. - a_x * std::pow(d_x, 2);
                }
                else if (d_x <= (w_v / 2 + w_b_ * 2))
                {
                    double b_x = 2. / (w_b_ * (w_v + 4 * w_b_));
                    p_x = b_x * std::pow((d_x - w_v / 2. - 2. * w_b_), 2);
                }
                if (d_y <= (h_v / 2 + h_b_))
                {
                    double a_y = 4. / ((h_v + 2. * h_b_) * (h_v + 4. * h_b_));
                    p_y = 1. - a_y * std::pow(d_y, 2);
                }
                else if (d_y <= (h_v / 2 + h_b_ * 2))
                {
                    double b_y = 2. / (h_b_ * (h_v + 4 * h_b_));
                    p_y = b_y * std::pow((d_y - h_v / 2. - h_b_ * 2.), 2);
                }
                bin_index_to_pxpy[bin_index] = p_x * p_y;
                cell_sum_pxpy += bin_index_to_pxpy[bin_index];
            }
        }
        for (const std::pair<const int, double> &bin_index_and_pxpy : bin_index_to_pxpy)
        {
            int bin_index = bin_index_and_pxpy.first;
            double pxpy = bin_index_and_pxpy.second;
            double addition = (pxpy / cell_sum_pxpy) * cur_module.area();
            #pragma omp atomic
            bin_density_minus_movable_area_[bin_index] += addition;
        }
    }
    for (const double &bin_density: bin_density_minus_movable_area_)
    {
        value_ += std::pow(bin_density, 2);
    }
    input_ = input;
    return value_;
}

const std::vector<Point2<double>> &Density::Backward()
{
    const size_t &kNumModule = placement_.numModules();
    // grad_ = std::vector<Point2<double>>(kNumModule);
    grad_.assign(kNumModule, Point2<double>(0.0, 0.0));
    
    #pragma omp parallel for
    for (size_t k = 0; k < kNumModule; ++k)
    {
        Module &cur_module = placement_.module(k);
        if (cur_module.isFixed())
        {
            continue;
        }
        const double &w_v = cur_module.width();
        const double &h_v = cur_module.height();
        double cell_sum_px_grad_py = 0., cell_sum_px_py_grad = 0., cell_sum_pxpy = 0.;
        std::map<int, double> bin_index_to_pxpy, bin_index_to_px_grad_py, bin_index_to_px_py_grad;
        // traverse through the bins
        for (const int &bin_index : module_to_bin_indexes_[k])
        {
            int i = bin_index / grid_num_;
            int j = bin_index % grid_num_;
            double bin_center_x = placement_.boundryLeft() + (i + 0.5) * w_b_;
            double bin_center_y = placement_.boundryBottom() + (j + 0.5) * h_b_;
            double p_x = 0., p_y = 0.;
            double d_x = std::abs(input_[k].x - bin_center_x);
            double d_y = std::abs(input_[k].y - bin_center_y);
            double p_x_grad = 0., p_y_grad = 0.;
            if (d_x <= (w_v / 2 + w_b_))
            {
                double a_x = 4. / ((w_v + 2. * w_b_) * (w_v + 4. * w_b_));
                p_x = 1. - a_x * std::pow(d_x, 2);
                p_x_grad = -2. * a_x * d_x;
                p_x_grad = (input_[k].x < bin_center_x) ? -p_x_grad : p_x_grad; // adjust gradient direction
            }
            else if (d_x <= (w_v / 2 + w_b_ * 2))
            {
                double b_x = 2. / (w_b_ * (w_v + 4 * w_b_));
                p_x = b_x * std::pow((d_x - w_v / 2. - 2. * w_b_), 2);
                p_x_grad = 2. * b_x * (d_x - w_v / 2. - w_b_ * 2.);
                // p_x_grad = (input_[k].x < (bin_center_x + w_v / 2. + w_b_ / 2.)) ? -p_x_grad : p_x_grad; // adjust gradient direction
                p_x_grad = (input_[k].x < bin_center_x) ? -p_x_grad : p_x_grad;
            }
            if (d_y <= (h_v / 2 + h_b_))
            {
                double a_y = 4. / ((h_v + 2. * h_b_) * (h_v + 4. * h_b_));
                p_y = 1. - a_y * std::pow(d_y, 2);
                p_y_grad = -2. * a_y * d_y;
                p_y_grad = (input_[k].y < bin_center_y) ? -p_y_grad : p_y_grad; // adjust gradient direction
            }
            else if (d_y <= (h_v / 2 + h_b_ * 2))
            {
                double b_y = 2. / (h_b_ * (h_v + 4 * h_b_));
                p_y = b_y * std::pow(d_y - h_v / 2. - h_b_ * 2., 2);
                p_y_grad = 2. * b_y * (d_y - h_v / 2. - h_b_ * 2.);
                // p_y_grad = (input_[k].y < (bin_center_y + h_v / 2. + h_b_ / 2.)) ? -p_y_grad : p_y_grad; // adjust gradient direction
                p_y_grad = (input_[k].y < bin_center_y) ? -p_y_grad : p_y_grad;
            }
            bin_index_to_pxpy[bin_index] = p_x * p_y;
            bin_index_to_px_py_grad[bin_index] = p_x * p_y_grad;
            bin_index_to_px_grad_py[bin_index] = p_x_grad * p_y;
            cell_sum_pxpy += bin_index_to_pxpy[bin_index];
            cell_sum_px_py_grad += bin_index_to_px_py_grad[bin_index];
            cell_sum_px_grad_py += bin_index_to_px_grad_py[bin_index];
        }
        for (const auto &bin_index_and_pxpy : bin_index_to_pxpy)
        {
            int bin_index = bin_index_and_pxpy.first;
            double pxpy = bin_index_and_pxpy.second;
            double px_grad_py = bin_index_to_px_grad_py[bin_index];
            double px_py_grad = bin_index_to_px_py_grad[bin_index];
            // assert(cell_sum_pxpy != 0.);
            double grad_x = 2 * bin_density_minus_movable_area_[bin_index] * cur_module.area() * (cell_sum_pxpy * px_grad_py - cell_sum_px_grad_py * pxpy) / pow(cell_sum_pxpy, 2);
            double grad_y = 2 * bin_density_minus_movable_area_[bin_index] * cur_module.area() * (cell_sum_pxpy * px_py_grad - cell_sum_px_py_grad * pxpy) / pow(cell_sum_pxpy, 2);
            #pragma omp atomic
            grad_[k].x += grad_x;
            #pragma omp atomic
            grad_[k].y += grad_y;
        }
    }
    return grad_;
}

ObjectiveFunction::ObjectiveFunction(Placement &placement, double grid_num, Wirelength &wirelength, Density &density)
    : BaseFunction(placement.numModules()), placement_(placement), grid_num_(grid_num), wirelength_(wirelength), density_(density), lambda_(1.0)
{
    printf("ObjectiveFunction initialized with grid_num = %.2f\n", grid_num);
}

const double &ObjectiveFunction::operator()(const std::vector<Point2<double>> &input)
{
    value_ = wirelength_(input) + lambda_ *  density_(input);
    return value_;
}

const std::vector<Point2<double>> &ObjectiveFunction::Backward()
{
    // Compute gradients for wirelength and density
    const std::vector<Point2<double>> &wirelength_grad = wirelength_.Backward();
    const std::vector<Point2<double>> &density_grad = density_.Backward();

    grad_ = wirelength_grad;
    for (size_t i = 0; i < grad_.size(); ++i)
    {
        grad_[i] += density_grad[i] * lambda_;
    }
    return grad_;
}