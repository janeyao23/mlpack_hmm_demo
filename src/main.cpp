#include <iostream>
#include <vector>
#include <mlpack/core.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/hmm/discrete_distribution.hpp>

/*
 * A demonstration program that uses mlpack's Hidden Markov Model (HMM) class
 * to illustrate the basic components of a discrete HMM:
 *   - the initial probability vector,
 *   - the state transition probability matrix,
 *   - and the emission probability distributions for each hidden state.
 *
 * The program constructs a simple two‑state HMM with discrete observations.
 * It prints the initial state probabilities, transition matrix and emission
 * probabilities, then uses the Viterbi algorithm to find the most likely
 * sequence of hidden states for a given observation sequence.  It also
 * computes the log‑likelihood of the observation sequence.  Finally, the
 * program retrains the HMM on the observed sequence using Baum–Welch and
 * prints the updated parameters.
 *
 * To compile this program you need to have mlpack and its dependencies (such
 * as Armadillo) available.  See the CMakeLists.txt and the GitHub Actions
 * workflow for details on how to build mlpack from source and compile this
 * example using Visual C++.
 */

using namespace std;
using namespace mlpack;
using namespace mlpack::distribution;

int main()
{
    // Number of hidden states in the HMM.
    const size_t states = 2;

    // Create emission distributions for each hidden state.  The
    // DiscreteDistribution type stores the probability of each observation
    // symbol.  Here each state can emit one of two symbols: 0 or 1.
    std::vector<DiscreteDistribution> emissions;
    emissions.reserve(states);
    emissions.emplace_back(/* number of symbols */ 2);
    emissions.emplace_back(/* number of symbols */ 2);

    // Define the emission probability vectors for each state.
    arma::vec probsState0(2);
    arma::vec probsState1(2);

    // In this example the first state (index 0) mostly emits the symbol 0.
    probsState0[0] = 0.9; // P(observation=0 | state=0)
    probsState0[1] = 0.1; // P(observation=1 | state=0)

    // The second state (index 1) mostly emits the symbol 1.
    probsState1[0] = 0.2; // P(observation=0 | state=1)
    probsState1[1] = 0.8; // P(observation=1 | state=1)

    // Assign the emission probabilities to the distributions.
    emissions[0].Probabilities() = probsState0;
    emissions[1].Probabilities() = probsState1;

    // Define the initial probability vector.  Each element is the probability
    // that the HMM starts in that state at time 0.  The vector should sum to 1.
    arma::vec initial(states);
    initial[0] = 0.5; // P(state=0 at t=0)
    initial[1] = 0.5; // P(state=1 at t=0)

    // Define the state transition matrix.  T(i, j) is the probability of
    // transitioning to state i from state j.  Each column of this matrix
    // should sum to 1.
    arma::mat transition(states, states);

    // From state 0: 80% probability of staying in state 0, 20% of switching to state 1.
    transition(0, 0) = 0.8; // P(0 -> 0)
    transition(1, 0) = 0.2; // P(1 -> 0)

    // From state 1: 30% probability of switching back to state 0, 70% of staying in state 1.
    transition(0, 1) = 0.3; // P(0 -> 1)
    transition(1, 1) = 0.7; // P(1 -> 1)

    // Construct the HMM with the specified parameters.  The last argument is
    // the tolerance for the Baum–Welch algorithm; we use the default value.
    HMM<DiscreteDistribution> hmm(initial, transition, emissions);

    // Print the model parameters.
    cout << "Initial state probabilities:" << endl;
    cout << hmm.Initial().t() << endl;

    cout << "State transition matrix:" << endl;
    cout << hmm.Transition() << endl;

    cout << "Emission probabilities for each state:" << endl;
    for (size_t s = 0; s < states; ++s)
    {
        cout << "  State " << s << ": " << hmm.Emission()[s].Probabilities().t();
    }
    cout << endl;

    // Define an observation sequence.  The type Row<size_t> stores a sequence
    // of discrete symbols.  Each element is the observed symbol at that time step.
    arma::Row<size_t> observations;
    observations = { 0, 0, 1, 0, 1, 1 };

    cout << "Observation sequence: ";
    for (size_t i = 0; i < observations.n_elem; ++i)
        cout << observations[i] << " ";
    cout << endl;

    // Compute the most likely sequence of hidden states using the Viterbi algorithm.
    arma::Row<size_t> predictedStates;
    hmm.Predict(observations, predictedStates);

    cout << "Predicted hidden states (Viterbi): ";
    for (size_t i = 0; i < predictedStates.n_elem; ++i)
        cout << predictedStates[i] << " ";
    cout << endl;

    // Compute the log‑likelihood of the observation sequence under the model.
    double logLikelihood = hmm.LogLikelihood(observations);
    cout << "Log‑likelihood of observation sequence: " << logLikelihood << endl;

    // Retrain the model using the observation sequence.  We convert the discrete
    // observations into a 1×N matrix of doubles because the unsupervised
    // Train() method expects each observation sequence as a matrix with the
    // number of rows equal to the dimensionality (here 1) and columns equal
    // to the number of observations.  Note that Baum–Welch will re‑estimate
    // the initial probabilities, transition matrix and emission distributions.
    arma::mat trainSeq(1, observations.n_elem);
    for (size_t i = 0; i < observations.n_elem; ++i)
    {
        trainSeq(0, i) = static_cast<double>(observations[i]);
    }
    std::vector<arma::mat> trainingData;
    trainingData.push_back(trainSeq);
    hmm.Train(trainingData);

    // Display the learned parameters.
    cout << endl << "Parameters after Baum–Welch training:" << endl;
    cout << "Updated initial state probabilities:" << endl;
    cout << hmm.Initial().t() << endl;

    cout << "Updated transition matrix:" << endl;
    cout << hmm.Transition() << endl;

    cout << "Updated emission probabilities:" << endl;
    for (size_t s = 0; s < states; ++s)
    {
        cout << "  State " << s << ": " << hmm.Emission()[s].Probabilities().t();
    }
    cout << endl;

    return 0;
}
