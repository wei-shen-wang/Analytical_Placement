#include "GlobalPlacer.h"

#include <cstdio>
#include <vector>

#include "ObjectiveFunction.h"
#include "Optimizer.h"
#include "Point.h"

GlobalPlacer::GlobalPlacer(Placement &placement)
    : _placement(placement)
{
}

void GlobalPlacer::place()
{
    ////////////////////////////////////////////////////////////////////
    // This section is an example for analytical methods.
    ////////////////////////////////////////////////////////////////////
    // Global placement algorithm
    ////////////////////////////////////////////////////////////////////
    // Place all module in the center of the placement area.
    double chip_mid_x = _placement.rectangleChip().centerX();
    double chip_mid_y = _placement.rectangleChip().centerY();
    const size_t &kNumModule = _placement.numModules();
    double grid_num = std::pow(kNumModule, 0.5);
    double w_b = (_placement.boundryRight() - _placement.boundryLeft()) / grid_num;
    double h_b = (_placement.boundryTop() - _placement.boundryBottom()) / grid_num;
    srand(1); // Seed the random number generator for reproducibility
    std::vector<Point2<double>> positions(kNumModule); // Optimization variables (positions of modules). You may modify this line.
    for (size_t i = 0; i < kNumModule; i++)
    {
        Module &module = _placement.module(i);
        if (module.isFixed())
        {
            positions[i] = Point2<double>(module.centerX(), module.centerY());
            continue; // Skip fixed modules.
        }
        positions[i] = Point2<double>(chip_mid_x, chip_mid_y);
    }
    Wirelength wirelength(_placement); // Wirelength objective function
    SimpleConjugateGradient wl_optimizer(wirelength, positions, grid_num, _placement); // Wirelength optimizer
    wl_optimizer.setScalingFactor(0.1); // Set the scaling factor for the step
    for (int i = 0;i < 1000; i++)
    {
        wl_optimizer.Step(); // Perform one optimization step
        double wl = wirelength(positions); // Compute the wirelength
        if (i % 100 == 0)
        {
            printf("wl = %e, alpha = %.2f\n", wl, wl_optimizer.getAlpha());
            fflush(stdout);
        }
    }
    // for (size_t i = 0; i < kNumModule; i++)
    // {
    //     Module &module = _placement.module(i);
    //     if (module.isFixed())
    //     {
    //         continue; // Skip fixed modules.
    //     }
    //     positions[i].x *= 2.5;
    // }
    Density density(_placement, grid_num);
    wirelength(positions); // Compute the wirelength
    wirelength.Backward(); // Compute the wirelength gradient
    double partialDerivativeWirelength = wirelength.getSumOfNormPartialDerivative();
    printf("partialDerivativeWirelength = %e\n", partialDerivativeWirelength);
    density(positions);
    density.Backward(); // Compute the density gradient
    double partialDerivativeDensity = density.getSumOfNormPartialDerivative();
    printf("partialDerivativeDensity = %e\n", partialDerivativeDensity);
    double lambda = partialDerivativeWirelength / partialDerivativeDensity; // Set the penalty weight for density
    printf("initial lambda = %e\n", lambda);
    double overflow_ratio = 100.0;      // Initialize overflow ratio
    constexpr double limit_overflow_ratio = 0.05; // Set a limit for the overflow ratio
    int kMaxInnerIter = 1000;
    constexpr int limit_no_improvement = 10; // Limit for no improvement iterations
    constexpr int limit_no_improvement_in_cg = 100; // Limit for no improvement in conjugate gradient iterations
    double best_overflow_ratio = 100.0; // Best overflow ratio found so far
    int iter = 0;
    int no_improvement = 0; // Counter for no improvement in iterations
    ObjectiveFunction objectiveFunction(_placement, grid_num, wirelength, density);
    SimpleConjugateGradient optimizer(objectiveFunction, positions, grid_num, _placement); // Optimizer
    optimizer.setScalingFactor(0.2);
    while ((overflow_ratio > limit_overflow_ratio) && (no_improvement < limit_no_improvement))
    { // Continue until the overflow ratio is small enough
        objectiveFunction.setLambda(lambda); // Set the penalty weight for density in the objective function
        optimizer.Initialize();
        double best_objective_value = objectiveFunction(positions); // Compute the initial objective value
        double objective_value = best_objective_value; // Initialize objective value
        double inner_best_overflow_ratio = density.calculateOverflowRatio();
        int no_improvement_in_cg = 0; // Counter for no improvement in conjugate gradient iterations
        std::vector<Point2<double>> best_positions = positions; // Store the best positions found so far
        int sub_iter = 0; // Counter for sub-iterations
        while ((no_improvement_in_cg < limit_no_improvement_in_cg) && (sub_iter < kMaxInnerIter)) // Perform optimization steps
        { // Perform optimization until the objective value stops improving
            optimizer.Step();
            overflow_ratio = density.calculateOverflowRatio();
            if (iter % 10 == 0){
                printf("iter = %d, sub_iter = %d, overflow = %.4f, f = %e, alpha = %.2f, lambda = %e\n", 
                       iter, sub_iter, overflow_ratio, objective_value, optimizer.getAlpha() ,lambda);
                fflush(stdout);
            }
            objective_value = objectiveFunction(positions); // Compute the objective value
            if (objective_value < best_objective_value)
            {
                best_objective_value = objective_value; // Update the best objective value
                best_positions = positions; // Update the best positions
                no_improvement_in_cg = 0; // Reset the no improvement counter
                inner_best_overflow_ratio = overflow_ratio;
            }
            else
            {
                no_improvement_in_cg++; // Increment the no improvement counter
            }
            ++sub_iter;
            ++iter;
        }
        positions = best_positions; // Update positions to the best found
        if (inner_best_overflow_ratio < best_overflow_ratio)
        {
            best_overflow_ratio = inner_best_overflow_ratio; // Update the best overflow ratio
            no_improvement = 0;
        }
        else{
            no_improvement++;
        }
        lambda *= 2;
        kMaxInnerIter = max(500, static_cast<int>(kMaxInnerIter*0.9));
    }
    ////////////////////////////////////////////////////////////////////
    // Write the placement result into the database. (You may modify this part.)
    ////////////////////////////////////////////////////////////////////
    size_t fixed_cnt = 0;
    for (size_t i = 0; i < kNumModule; i++)
    {
        // If the module is fixed, its position should not be changed.
        // In this programing assignment, a fixed module may be a terminal or a pre-placed module.
        if (_placement.module(i).isFixed())
        {
            fixed_cnt++;
            continue;
        }
        _placement.module(i).setCenterPosition(positions[i].x, positions[i].y);
    }
    printf("INFO: %lu / %lu modules are fixed.\n", fixed_cnt, kNumModule);
}

void GlobalPlacer::plotPlacementResult(const string outfilename, bool isPrompt)
{
    ofstream outfile(outfilename.c_str(), ios::out);
    outfile << " " << endl;
    outfile << "set title \"wirelength = " << _placement.computeHpwl() << "\"" << endl;
    outfile << "set size ratio 1" << endl;
    outfile << "set nokey" << endl
            << endl;
    outfile << "plot[:][:] '-' w l lt 3 lw 2, '-' w l lt 1" << endl
            << endl;
    outfile << "# bounding box" << endl;
    plotBoxPLT(outfile, _placement.boundryLeft(), _placement.boundryBottom(), _placement.boundryRight(), _placement.boundryTop());
    outfile << "EOF" << endl;
    outfile << "# modules" << endl
            << "0.00, 0.00" << endl
            << endl;
    for (size_t i = 0; i < _placement.numModules(); ++i)
    {
        Module &module = _placement.module(i);
        plotBoxPLT(outfile, module.x(), module.y(), module.x() + module.width(), module.y() + module.height());
    }
    outfile << "EOF" << endl;
    outfile << "pause -1 'Press any key to close.'" << endl;
    outfile.close();

    if (isPrompt)
    {
        char cmd[200];
        sprintf(cmd, "gnuplot %s", outfilename.c_str());
        if (!system(cmd))
        {
            cout << "Fail to execute: \"" << cmd << "\"." << endl;
        }
    }
}

void GlobalPlacer::plotBoxPLT(ofstream &stream, double x1, double y1, double x2, double y2)
{
    stream << x1 << ", " << y1 << endl
           << x2 << ", " << y1 << endl
           << x2 << ", " << y2 << endl
           << x1 << ", " << y2 << endl
           << x1 << ", " << y1 << endl
           << endl;
}
