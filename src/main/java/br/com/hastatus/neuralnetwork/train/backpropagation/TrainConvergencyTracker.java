package br.com.hastatus.neuralnetwork.train.backpropagation;

public class TrainConvergencyTracker {

    double meanSquareError;

    public void addMeanSquareError(double error) {
        meanSquareError += Math.pow(error, 2);
    }

}
