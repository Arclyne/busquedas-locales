#include <algorithm>
#include <atomic>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

constexpr double LOWER_BOUND = -10.0;
constexpr double UPPER_BOUND = 10.0;
const std::vector<int> DIMENSIONS = {10, 30, 50, 100};

// Budget and stopping criterion of the CEC 2005 benchmark: MaxFES = 10 000 · D, success when f(x) <= 1e-8.
constexpr long EVALUATIONS_PER_DIMENSION = 10000;
constexpr double TARGET_ERROR = 1e-8;

constexpr int NUMBER_OF_RUNS = 30;
constexpr unsigned int BASE_SEED = 1000;
constexpr int CHECKPOINTS = 100;

// Parameter tuning: every configuration of the grid runs TUNING_RUNS times with seeds
// that are not used by the final experiment.
constexpr int TUNING_RUNS = 5;
constexpr unsigned int TUNING_BASE_SEED = 5000;
const std::vector<double> EXPECTED_MUTATED_GENES = {1, 2, 5, 10};          // p = k / D
const std::vector<double> STEP_SIZES = {0.001, 0.01, 0.1, 1.0};            // r
const std::vector<int> TWEAKS_PER_ITERATION = {2, 5, 10, 20, 50};          // n
const std::vector<double> INITIAL_TEMPERATURES = {0.001, 0.01, 0.1, 1.0, 10.0}; // T0
constexpr double FINAL_TEMPERATURE = 1e-10;

const std::string RESULTS_DIRECTORY = "results";

enum class Algorithm {
    HillClimbing,
    SteepestAscent,
    SteepestAscentWithReplacement,
    SimulatedAnnealing
};

const std::vector<Algorithm> ALGORITHMS = {
    Algorithm::HillClimbing,
    Algorithm::SteepestAscent,
    Algorithm::SteepestAscentWithReplacement,
    Algorithm::SimulatedAnnealing
};

struct Parameters {
    double expectedMutatedGenes;
    double alterationProbability;
    double stepSize;
    int tweaks;
    double initialTemperature;
};

struct RunResult {
    double bestFx;
    long evaluations;
    double milliseconds;
    std::vector<double> trace;
};

struct Statistics {
    double best;
    double worst;
    double mean;
    double median;
    double standardDeviation;
};

std::string algorithmName(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::HillClimbing: return "HillClimbing";
        case Algorithm::SteepestAscent: return "SteepestAscent";
        case Algorithm::SteepestAscentWithReplacement: return "SteepestAscentWithReplacement";
        case Algorithm::SimulatedAnnealing: return "SimulatedAnnealing";
    }
    return "";
}

double sphere(const std::vector<double>& x) {
    double total = 0.0;
    for (double value : x) {
        total += value * value;
    }
    return total;
}

// Counts every call to the objective function, remembers the best value seen so far and
// stores it at evenly spaced checkpoints to draw convergence curves.
class Evaluator {
public:
    Evaluator(long maxEvaluations, bool recordTrace)
        : maxEvaluations_(maxEvaluations), recordTrace_(recordTrace) {
        if (recordTrace_) {
            trace_.resize(CHECKPOINTS + 1);
        }
    }

    double evaluate(const std::vector<double>& x) {
        double fx = sphere(x);
        ++evaluations_;
        bestFx_ = std::min(bestFx_, fx);

        while (recordTrace_ && nextCheckpoint_ <= CHECKPOINTS && evaluations_ >= checkpointEvaluation(nextCheckpoint_)) {
            trace_[nextCheckpoint_++] = bestFx_;
        }
        return fx;
    }

    bool finished() const {
        return evaluations_ >= maxEvaluations_ || bestFx_ <= TARGET_ERROR;
    }

    long maxEvaluations() const { return maxEvaluations_; }
    long evaluations() const { return evaluations_; }
    double bestFx() const { return bestFx_; }

    std::vector<double> trace() {
        while (recordTrace_ && nextCheckpoint_ <= CHECKPOINTS) {
            trace_[nextCheckpoint_++] = bestFx_;
        }
        return trace_;
    }

    long checkpointEvaluation(int checkpoint) const {
        return checkpoint == 0 ? 1 : checkpoint * maxEvaluations_ / CHECKPOINTS;
    }

private:
    long maxEvaluations_;
    bool recordTrace_;
    long evaluations_ = 0;
    double bestFx_ = INFINITY;
    int nextCheckpoint_ = 0;
    std::vector<double> trace_;
};

std::vector<double> randomSolution(int dimensions, std::mt19937& generator) {
    std::uniform_real_distribution<double> valueDistribution(LOWER_BOUND, UPPER_BOUND);
    std::vector<double> solution(dimensions);
    for (double& value : solution) {
        value = valueDistribution(generator);
    }
    return solution;
}

// Bounded Uniform Convolution (Luke, Algorithm 8). The vector arrives by value: Tweak(Copy(S)).
std::vector<double> tweak(std::vector<double> v, const Parameters& parameters, std::mt19937& generator) {
    std::uniform_real_distribution<double> probabilityDistribution(0.0, 1.0);
    std::uniform_real_distribution<double> noiseDistribution(-parameters.stepSize, parameters.stepSize);

    for (double& value : v) {
        if (parameters.alterationProbability >= probabilityDistribution(generator)) {
            double noise;
            do {
                noise = noiseDistribution(generator);
            } while (value + noise < LOWER_BOUND || value + noise > UPPER_BOUND);
            value += noise;
        }
    }
    return v;
}

// Hill-Climbing (Luke, Algorithm 4).
double hillClimbing(int dimensions, const Parameters& parameters, Evaluator& evaluator, std::mt19937& generator) {
    std::vector<double> s = randomSolution(dimensions, generator);
    double sFx = evaluator.evaluate(s);

    while (!evaluator.finished()) {
        std::vector<double> r = tweak(s, parameters, generator);
        double rFx = evaluator.evaluate(r);
        if (rFx < sFx) {
            s = std::move(r);
            sFx = rFx;
        }
    }
    return sFx;
}

// Samples n tweaks of S and returns the best one; stops early when the budget runs out.
std::vector<double> bestOfTweaks(const std::vector<double>& s, const Parameters& parameters, Evaluator& evaluator,
                                 std::mt19937& generator, double& bestFx) {
    std::vector<double> r = tweak(s, parameters, generator);
    bestFx = evaluator.evaluate(r);

    for (int sample = 1; sample < parameters.tweaks && !evaluator.finished(); ++sample) {
        std::vector<double> w = tweak(s, parameters, generator);
        double wFx = evaluator.evaluate(w);
        if (wFx < bestFx) {
            r = std::move(w);
            bestFx = wFx;
        }
    }
    return r;
}

// Steepest Ascent Hill-Climbing (Luke, Algorithm 5).
double steepestAscent(int dimensions, const Parameters& parameters, Evaluator& evaluator, std::mt19937& generator) {
    std::vector<double> s = randomSolution(dimensions, generator);
    double sFx = evaluator.evaluate(s);

    while (!evaluator.finished()) {
        double rFx;
        std::vector<double> r = bestOfTweaks(s, parameters, evaluator, generator, rFx);
        if (rFx < sFx) {
            s = std::move(r);
            sFx = rFx;
        }
    }
    return sFx;
}

// Steepest Ascent Hill-Climbing With Replacement (Luke, Algorithm 6).
double steepestAscentWithReplacement(int dimensions, const Parameters& parameters, Evaluator& evaluator,
                                     std::mt19937& generator) {
    std::vector<double> s = randomSolution(dimensions, generator);
    double sFx = evaluator.evaluate(s);
    std::vector<double> best = s;
    double bestFx = sFx;

    while (!evaluator.finished()) {
        s = bestOfTweaks(s, parameters, evaluator, generator, sFx);
        if (sFx < bestFx) {
            best = s;
            bestFx = sFx;
        }
    }
    return bestFx;
}

// Simulated Annealing (Luke, Algorithm 13) with geometric cooling from T0 to FINAL_TEMPERATURE
// over the whole evaluation budget.
double simulatedAnnealing(int dimensions, const Parameters& parameters, Evaluator& evaluator, std::mt19937& generator) {
    std::uniform_real_distribution<double> probabilityDistribution(0.0, 1.0);
    double temperature = parameters.initialTemperature;
    const double coolingRate = std::pow(FINAL_TEMPERATURE / parameters.initialTemperature,
                                        1.0 / static_cast<double>(evaluator.maxEvaluations() - 1));

    std::vector<double> s = randomSolution(dimensions, generator);
    double sFx = evaluator.evaluate(s);
    std::vector<double> best = s;
    double bestFx = sFx;

    while (!evaluator.finished()) {
        std::vector<double> r = tweak(s, parameters, generator);
        double rFx = evaluator.evaluate(r);
        if (rFx < sFx || probabilityDistribution(generator) < std::exp((sFx - rFx) / temperature)) {
            s = std::move(r);
            sFx = rFx;
        }
        temperature *= coolingRate;
        if (sFx < bestFx) {
            best = s;
            bestFx = sFx;
        }
    }
    return bestFx;
}

RunResult runAlgorithm(Algorithm algorithm, int dimensions, const Parameters& parameters, unsigned int seed,
                       bool recordTrace) {
    std::mt19937 generator(seed);
    Evaluator evaluator(EVALUATIONS_PER_DIMENSION * dimensions, recordTrace);

    // CPU time of the process: unlike wall-clock time it does not count the moments the OS gives the core to
    // another process.
    std::clock_t start = std::clock();
    double bestFx = 0.0;
    switch (algorithm) {
        case Algorithm::HillClimbing:
            bestFx = hillClimbing(dimensions, parameters, evaluator, generator);
            break;
        case Algorithm::SteepestAscent:
            bestFx = steepestAscent(dimensions, parameters, evaluator, generator);
            break;
        case Algorithm::SteepestAscentWithReplacement:
            bestFx = steepestAscentWithReplacement(dimensions, parameters, evaluator, generator);
            break;
        case Algorithm::SimulatedAnnealing:
            bestFx = simulatedAnnealing(dimensions, parameters, evaluator, generator);
            break;
    }
    std::clock_t end = std::clock();

    double milliseconds = 1000.0 * static_cast<double>(end - start) / CLOCKS_PER_SEC;
    return {bestFx, evaluator.evaluations(), milliseconds, evaluator.trace()};
}

std::vector<Parameters> parameterGrid(Algorithm algorithm, int dimensions) {
    std::vector<Parameters> grid;
    for (double genes : EXPECTED_MUTATED_GENES) {
        double probability = std::min(1.0, genes / dimensions);
        for (double stepSize : STEP_SIZES) {
            switch (algorithm) {
                case Algorithm::HillClimbing:
                    grid.push_back({genes, probability, stepSize, 1, 0.0});
                    break;
                case Algorithm::SteepestAscent:
                case Algorithm::SteepestAscentWithReplacement:
                    for (int tweaks : TWEAKS_PER_ITERATION) {
                        grid.push_back({genes, probability, stepSize, tweaks, 0.0});
                    }
                    break;
                case Algorithm::SimulatedAnnealing:
                    for (double temperature : INITIAL_TEMPERATURES) {
                        grid.push_back({genes, probability, stepSize, 1, temperature});
                    }
                    break;
            }
        }
    }
    return grid;
}

Statistics computeStatistics(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t count = values.size();

    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    double mean = sum / count;

    double squaredDeviations = 0.0;
    for (double value : values) {
        squaredDeviations += (value - mean) * (value - mean);
    }
    double standardDeviation = count > 1 ? std::sqrt(squaredDeviations / (count - 1)) : 0.0;

    double median = count % 2 == 1 ? values[count / 2] : (values[count / 2 - 1] + values[count / 2]) / 2.0;
    return {values.front(), values.back(), mean, median, standardDeviation};
}

struct TuningScore {
    Parameters parameters;
    int successes;
    double medianFx;
    double meanEvaluations;
};

// Lexicographic order: more successes, then lower median of f(x) clipped at the target, then fewer evaluations.
bool isBetterScore(const TuningScore& first, const TuningScore& second) {
    if (first.successes != second.successes) {
        return first.successes > second.successes;
    }
    double firstMedian = std::max(first.medianFx, TARGET_ERROR);
    double secondMedian = std::max(second.medianFx, TARGET_ERROR);
    if (firstMedian != secondMedian) {
        return firstMedian < secondMedian;
    }
    return first.meanEvaluations < second.meanEvaluations;
}

struct TuningTask {
    size_t algorithmIndex;
    size_t dimensionIndex;
    size_t configurationIndex;
    int run;
};

// selected[algorithmIndex][dimensionIndex]
std::vector<std::vector<Parameters>> tuneParameters(unsigned int threads) {
    std::vector<std::vector<std::vector<Parameters>>> grids(ALGORITHMS.size());
    std::vector<std::vector<std::vector<std::vector<RunResult>>>> results(ALGORITHMS.size());
    std::vector<TuningTask> tasks;

    for (size_t a = 0; a < ALGORITHMS.size(); ++a) {
        for (size_t d = 0; d < DIMENSIONS.size(); ++d) {
            grids[a].push_back(parameterGrid(ALGORITHMS[a], DIMENSIONS[d]));
            results[a].push_back(std::vector<std::vector<RunResult>>(grids[a][d].size(), std::vector<RunResult>(TUNING_RUNS)));
            for (size_t c = 0; c < grids[a][d].size(); ++c) {
                for (int run = 0; run < TUNING_RUNS; ++run) {
                    tasks.push_back({a, d, c, run});
                }
            }
        }
    }

    // The most expensive tasks (largest dimension) go first so the threads finish together.
    std::stable_sort(tasks.begin(), tasks.end(), [](const TuningTask& first, const TuningTask& second) {
        return first.dimensionIndex > second.dimensionIndex;
    });

    std::cout << "Tuning: " << tasks.size() << " pilot runs on " << threads << " threads..." << std::endl;
    std::atomic<size_t> nextTask{0};
    std::vector<std::thread> workers;
    for (unsigned int t = 0; t < threads; ++t) {
        workers.emplace_back([&]() {
            for (size_t index = nextTask++; index < tasks.size(); index = nextTask++) {
                const TuningTask& task = tasks[index];
                const Parameters& parameters = grids[task.algorithmIndex][task.dimensionIndex][task.configurationIndex];
                results[task.algorithmIndex][task.dimensionIndex][task.configurationIndex][task.run] =
                    runAlgorithm(ALGORITHMS[task.algorithmIndex], DIMENSIONS[task.dimensionIndex], parameters,
                                 TUNING_BASE_SEED + task.run + 1, false);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    std::ofstream tuningFile(RESULTS_DIRECTORY + "/tuning.csv");
    tuningFile << "Algorithm,Dimensions,k,p,r,n,T0,Successes,MedianFx,MeanFx,MeanEvaluations\n";

    std::vector<std::vector<Parameters>> selected(ALGORITHMS.size());
    for (size_t a = 0; a < ALGORITHMS.size(); ++a) {
        for (size_t d = 0; d < DIMENSIONS.size(); ++d) {
            TuningScore bestScore{};
            bool hasBest = false;

            for (size_t c = 0; c < grids[a][d].size(); ++c) {
                std::vector<double> fxValues;
                double evaluationSum = 0.0;
                int successes = 0;
                for (const RunResult& result : results[a][d][c]) {
                    fxValues.push_back(result.bestFx);
                    evaluationSum += result.evaluations;
                    successes += result.bestFx <= TARGET_ERROR ? 1 : 0;
                }
                Statistics statistics = computeStatistics(fxValues);
                TuningScore score{grids[a][d][c], successes, statistics.median, evaluationSum / TUNING_RUNS};

                const Parameters& p = score.parameters;
                tuningFile << algorithmName(ALGORITHMS[a]) << ',' << DIMENSIONS[d] << ',' << p.expectedMutatedGenes << ','
                           << p.alterationProbability << ',' << p.stepSize << ',' << p.tweaks << ',' << p.initialTemperature
                           << ',' << successes << ',' << std::scientific << statistics.median << ',' << statistics.mean
                           << std::defaultfloat << ',' << score.meanEvaluations << '\n';

                if (!hasBest || isBetterScore(score, bestScore)) {
                    bestScore = score;
                    hasBest = true;
                }
            }
            selected[a].push_back(bestScore.parameters);
        }
    }
    return selected;
}

void writeSelectedParameters(const std::vector<std::vector<Parameters>>& selected) {
    std::ofstream parametersFile(RESULTS_DIRECTORY + "/parameters.csv");
    parametersFile << "Algorithm,Dimensions,k,p,r,n,T0\n";

    std::cout << "\nSelected parameters\n";
    std::cout << std::left << std::setw(32) << "Algorithm" << std::right << std::setw(6) << "D" << std::setw(6) << "k"
              << std::setw(10) << "p" << std::setw(8) << "r" << std::setw(6) << "n" << std::setw(8) << "T0" << '\n';

    for (size_t a = 0; a < ALGORITHMS.size(); ++a) {
        for (size_t d = 0; d < DIMENSIONS.size(); ++d) {
            const Parameters& p = selected[a][d];
            parametersFile << algorithmName(ALGORITHMS[a]) << ',' << DIMENSIONS[d] << ',' << p.expectedMutatedGenes << ','
                           << p.alterationProbability << ',' << p.stepSize << ',' << p.tweaks << ','
                           << p.initialTemperature << '\n';
            std::cout << std::left << std::setw(32) << algorithmName(ALGORITHMS[a]) << std::right << std::setw(6)
                      << DIMENSIONS[d] << std::setw(6) << p.expectedMutatedGenes << std::setw(10)
                      << p.alterationProbability << std::setw(8) << p.stepSize << std::setw(6) << p.tweaks
                      << std::setw(8) << p.initialTemperature << '\n';
        }
    }
}

void runExperiment(const std::vector<std::vector<Parameters>>& selected) {
    std::ofstream runsFile(RESULTS_DIRECTORY + "/runs.csv");
    std::ofstream summaryFile(RESULTS_DIRECTORY + "/summary.csv");
    std::ofstream convergenceFile(RESULTS_DIRECTORY + "/convergence.csv");
    runsFile << "Algorithm,Dimensions,Run,Seed,BestFx,Evaluations,TimeMs,Success\n";
    summaryFile << "Algorithm,Dimensions,Best,Worst,Mean,Median,StdDev,MeanTimeMs,StdTimeMs,"
                   "MeanEvaluations,StdEvaluations,Successes\n";
    convergenceFile << "Algorithm,Dimensions,Run,Checkpoint,Evaluations,BestFx\n";

    for (size_t d = 0; d < DIMENSIONS.size(); ++d) {
        const int dimensions = DIMENSIONS[d];
        std::cout << "\nD = " << dimensions << " (max evaluations: " << EVALUATIONS_PER_DIMENSION * dimensions << ")\n";
        std::cout << std::left << std::setw(32) << "Algorithm" << std::right << std::setw(13) << "Best"
                  << std::setw(13) << "Worst" << std::setw(13) << "Mean" << std::setw(13) << "Median" << std::setw(13)
                  << "StdDev" << std::setw(12) << "Time(ms)" << std::setw(13) << "Evaluations" << std::setw(10)
                  << "Success" << '\n';

        for (size_t a = 0; a < ALGORITHMS.size(); ++a) {
            const std::string name = algorithmName(ALGORITHMS[a]);
            const long maxEvaluations = EVALUATIONS_PER_DIMENSION * dimensions;
            std::vector<double> fxValues, times, evaluations;
            int successes = 0;

            for (int run = 1; run <= NUMBER_OF_RUNS; ++run) {
                const unsigned int seed = BASE_SEED + run;
                RunResult result = runAlgorithm(ALGORITHMS[a], dimensions, selected[a][d], seed, true);
                bool success = result.bestFx <= TARGET_ERROR;
                successes += success ? 1 : 0;
                fxValues.push_back(result.bestFx);
                times.push_back(result.milliseconds);
                evaluations.push_back(static_cast<double>(result.evaluations));

                runsFile << name << ',' << dimensions << ',' << run << ',' << seed << ',' << std::scientific
                         << std::setprecision(6) << result.bestFx << std::fixed << std::setprecision(3) << ','
                         << result.evaluations << ',' << result.milliseconds << ',' << (success ? 1 : 0) << '\n';

                for (int checkpoint = 0; checkpoint <= CHECKPOINTS; ++checkpoint) {
                    long checkpointEvaluations = checkpoint == 0 ? 1 : checkpoint * maxEvaluations / CHECKPOINTS;
                    convergenceFile << name << ',' << dimensions << ',' << run << ',' << checkpoint << ','
                                    << checkpointEvaluations << ',' << std::scientific << std::setprecision(6)
                                    << result.trace[checkpoint] << std::defaultfloat << '\n';
                }
            }

            Statistics fx = computeStatistics(fxValues);
            Statistics time = computeStatistics(times);
            Statistics evaluation = computeStatistics(evaluations);

            summaryFile << name << ',' << dimensions << ',' << std::scientific << std::setprecision(6) << fx.best << ','
                        << fx.worst << ',' << fx.mean << ',' << fx.median << ',' << fx.standardDeviation << std::fixed
                        << std::setprecision(3) << ',' << time.mean << ',' << time.standardDeviation << ','
                        << evaluation.mean << ',' << evaluation.standardDeviation << ',' << successes << '\n';

            std::cout << std::left << std::setw(32) << name << std::right << std::scientific << std::setprecision(3)
                      << std::setw(13) << fx.best << std::setw(13) << fx.worst << std::setw(13) << fx.mean
                      << std::setw(13) << fx.median << std::setw(13) << fx.standardDeviation << std::fixed
                      << std::setprecision(1) << std::setw(12) << time.mean << std::setw(13) << std::setprecision(0)
                      << evaluation.mean << std::setw(7) << successes << "/" << NUMBER_OF_RUNS << '\n';
        }
    }
}

int main(int argc, char* argv[]) {
    unsigned int threads = std::max(1u, std::thread::hardware_concurrency());
    if (argc > 1) {
        threads = static_cast<unsigned int>(std::max(1, std::stoi(argv[1])));
    }

    std::filesystem::create_directories(RESULTS_DIRECTORY);
    std::cout << "Sphere function, bounds [" << LOWER_BOUND << ", " << UPPER_BOUND << "], max evaluations "
              << EVALUATIONS_PER_DIMENSION << " * D, target f(x) <= " << TARGET_ERROR << ", runs " << NUMBER_OF_RUNS
              << "\n";

    std::vector<std::vector<Parameters>> selected = tuneParameters(threads);
    writeSelectedParameters(selected);

    // The final runs are sequential: std::clock measures the CPU time of the whole process.
    runExperiment(selected);

    std::cout << "\nResults saved to " << RESULTS_DIRECTORY << "/\n";
    return 0;
}
