# VLSI Physical Design Analytical Placement

| <img src ="/gifs/ibm01-cu85.gif"> | <img src ="/gifs/ibm09-cu90.gif"> |
|:---:|:---:|
| ibm01 | ibm09 |

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
| ibm01  | 46.65 |
| ibm02  | 131.84 |
| ibm05  | 9.37 |
| ibm07  | 302.94 |
| ibm08  | 318.37 |
| ibm09  | 273.66 |

## Reference

[1] Hsu, Meng-Kai, Valeriy Balabanov, and Yao-Wen Chang. "TSV-aware analytical placement for 3-D IC designs based on a novel weighted-average wirelength model." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 32.4 (2013): 497-509.

[2] Kahng, A. B., and Qinke Wang. "Implementation and extensibility of an analytic placer." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 24.5 (2006): 734-747.