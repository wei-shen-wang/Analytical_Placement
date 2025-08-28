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
    double original_center_x = _placement.rectangleChip().centerX();
    double original_center_y = _placement.rectangleChip().centerY();
    _placement.moveDesignCenter(-original_center_x, -original_center_y); // Move the design center to (0, 0)
    double chip_mid_x = _placement.rectangleChip().centerX();
    double chip_mid_y = _placement.rectangleChip().centerY();
    const size_t &kNumModule = _placement.numModules();
    double grid_num = std::sqrt(kNumModule);
    std::vector<Point2<double>> positions(kNumModule);
    for (size_t i = 0; i < kNumModule; i++)
    {
        Module &module = _placement.module(i);
        if (module.isFixed())
        {
            positions[i] = Point2<double>(module.centerX(), module.centerY());
            continue;
        }
        positions[i] = Point2<double>(chip_mid_x, chip_mid_y);
    }
    Wirelength wirelength(_placement);
    SimpleConjugateGradient wl_optimizer(wirelength, positions, grid_num, _placement);
    wl_optimizer.setScalingFactor(0.1);
    for (int i = 0;i < 1000; i++)
    {
        wl_optimizer.Step();
    }
    Density density(_placement, grid_num);
    wirelength(positions);
    wirelength.Backward();
    double partialDerivativeWirelength = wirelength.getSumOfNormPartialDerivative();
    density(positions);
    density.Backward();
    double partialDerivativeDensity = density.getSumOfNormPartialDerivative();
    double lambda = partialDerivativeWirelength / partialDerivativeDensity; // Set the penalty weight for density
    double overflow_ratio = 100.0;      // Initialize overflow ratio
    constexpr double limit_overflow_ratio = 0.01; // Set a limit for the overflow ratio
    int kMaxInnerIter = 1000;
    constexpr int limit_no_improvement_in_overflowratio = 10;
    constexpr int limit_no_improvement_in_cg = 100;
    double best_overflow_ratio = 100.0;
    int iter = 0;
    int no_improvement = 0; // Counter for no improvement in iterations
    ObjectiveFunction objectiveFunction(_placement, grid_num, wirelength, density);
    SimpleConjugateGradient optimizer(objectiveFunction, positions, grid_num, _placement);
    optimizer.setScalingFactor(0.2);
    while ((overflow_ratio > limit_overflow_ratio) && (no_improvement < limit_no_improvement_in_overflowratio))
    {
        objectiveFunction.setLambda(lambda);
        optimizer.Initialize();
        double best_objective_value = objectiveFunction(positions); // Compute the initial objective value
        double objective_value = best_objective_value; // Initialize objective value
        double inner_best_overflow_ratio = density.calculateOverflowRatio();
        int no_improvement_in_cg = 0; // Counter for no improvement in conjugate gradient iterations
        std::vector<Point2<double>> best_positions = positions;
        int sub_iter = 0;
        while ((no_improvement_in_cg < limit_no_improvement_in_cg) && (sub_iter < kMaxInnerIter))
        {
            optimizer.Step();
            overflow_ratio = density.calculateOverflowRatio();
            if (iter % 100 == 0){
                // for (size_t i = 0; i < kNumModule; i++)
                // {
                //     if (_placement.module(i).isFixed())
                //     {
                //         continue;
                //     }
                //     _placement.module(i).setCenterPosition(positions[i].x, positions[i].y);
                // }
                // _placement.moveDesignCenter(original_center_x, original_center_y);
                // string pltName = "./" + _placement.name() + "/placement_iter_" + to_string(iter) + ".plt";
                // plotPlacementResult(pltName, false);
                // _placement.moveDesignCenter(-original_center_x, -original_center_y);
                printf("iter = %d, sub_iter = %d, overflow = %.4f, f = %e, alpha = %.2f, lambda = %e, no improve = %d\n", 
                       iter, sub_iter, overflow_ratio, objective_value, optimizer.getAlpha() ,lambda, no_improvement_in_cg);
                fflush(stdout);
            }
            objective_value = objectiveFunction(positions); // Compute the objective value
            double imporovement = best_objective_value - objective_value;
            if (objective_value < best_objective_value)
            {
                best_objective_value = objective_value;
                best_positions = positions;
                inner_best_overflow_ratio = overflow_ratio;
                // no_improvement_in_cg = 0;
                no_improvement_in_cg = (imporovement < 1e-5) ? no_improvement_in_cg + 1 : 0;
            }
            else
            {
                no_improvement_in_cg++;
            }
            ++sub_iter;
            ++iter;
        }
        positions = best_positions;
        if (inner_best_overflow_ratio < best_overflow_ratio)
        {
            best_overflow_ratio = inner_best_overflow_ratio; // Update the best overflow ratio
            no_improvement = 0;
        }
        else{
            no_improvement++;
        }
        lambda *= 2.;
        kMaxInnerIter = max(500, static_cast<int>(kMaxInnerIter * 0.9));
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
    _placement.moveDesignCenter(original_center_x, original_center_y); // Move the design back
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
