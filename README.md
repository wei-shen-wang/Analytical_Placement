# VLSI Physical Design Analytical Placement

## Description

This project is an implementation of VLSI Placement using analytical method. Weighted-Average (WA) Wirelength Model [1] and Bell-Shape Potential Density Model[2] are implemented.

## File Structure

- benchmark: placement benchmark
- bin: executable binary
- lib: .a files
- results: placement result
- src: source code

## Usage

- Type the following to compile the source code
    ```
    make clean
    make
    ```

- Type the following to run the binary on all the benchmarks
    ```
    /runall.sh
    ```

## Experimental Results

| Benchmark | HPWL (1e6) |
|:---  | ---: |
| ibm01  | 47.71 |
| ibm02  | 131.94 |
| ibm05  | 9.48 |
| ibm07  | 321.84 |
| ibm08  | 318.50 |
| ibm09  | 281.34 |

## Reference

[1] Hsu, Meng-Kai, Valeriy Balabanov, and Yao-Wen Chang. "TSV-aware analytical placement for 3-D IC designs based on a novel weighted-average wirelength model." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 32.4 (2013): 497-509.

[2] Kahng, A. B., and Qinke Wang. "Implementation and extensibility of an analytic placer." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 24.5 (2006): 734-747.