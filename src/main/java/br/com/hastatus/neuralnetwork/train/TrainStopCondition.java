package br.com.hastatus.neuralnetwork.train;

/**
 * The {@code TrainStopCondition} class encapsulates the conditions under which the neural network training should be halted.
 *
 */
public class TrainStopCondition {

    private final int maxEpochs;
    private final double learningRate;
    private final boolean stopOnDiverge;
    private final double meanSquareErrorGood;


    /**
     * Constructs a new TrainStopCondition instance with specified training stop conditions.
     *
     * @param maxEpochs The maximum number of iterations over the entire training dataset to perform. More epochs can allow for more fine-tuned adjustments, but also increases the risk of overfitting.
     * @param learningRate The step size to use when adjusting weights and biases. Smaller values can lead to more precise adjustments but may require more epochs to converge to a solution.
     * @param stopOnDiverge A boolean flag that, when true, stops the training process if it is detected to be diverging, helping to prevent wasted computational resources on an unsuccessful training run.
     * @param meanSquareErrorGood The target mean square error value that is considered "good enough" for the training to stop. Once the training process achieves a mean square error less than or equal to this value, the training is halted.
     */
    public TrainStopCondition(int maxEpochs, double learningRate, boolean stopOnDiverge, double meanSquareErrorGood) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.stopOnDiverge = stopOnDiverge;
        this.meanSquareErrorGood = meanSquareErrorGood;

    }

    public int getMaxEpochs() {
        return maxEpochs;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public boolean isStopOnDiverge() {
        return stopOnDiverge;
    }

    public double getMeanSquareErrorGood() {
        return meanSquareErrorGood;
    }
}
