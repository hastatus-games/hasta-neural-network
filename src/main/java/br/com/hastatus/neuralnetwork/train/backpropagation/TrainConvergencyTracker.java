package br.com.hastatus.neuralnetwork.train.backpropagation;

import java.util.ArrayList;
import java.util.List;

public class TrainConvergencyTracker {

    double sumOfSquaredErrors;

    List<Double> meanSquareErrorsByEpoch;

    public TrainConvergencyTracker() {
        meanSquareErrorsByEpoch = new ArrayList<>();
    }

    /**
     * Add the difference between output value and expected to calculate the mean square error
     *
     * @param error The difference between output value and expected
     */
    public void addMeanSquareError(double error) {
        sumOfSquaredErrors += Math.pow(error, 2);
    }

    public void calculateAndSaveMeanSquareError(int samples, int outputs) {
        if (samples <= 0 || outputs <= 0) {
            throw new IllegalArgumentException("Number of samples and outputs must be greater than zero.");
        }

        double meanSquareError = (sumOfSquaredErrors / (samples * outputs));
        meanSquareErrorsByEpoch.add(meanSquareError);

        sumOfSquaredErrors = 0;
    }

    public double getMeanSquareErrorByEpoch(int epoch) {
        return meanSquareErrorsByEpoch.get(epoch);
    }
}

