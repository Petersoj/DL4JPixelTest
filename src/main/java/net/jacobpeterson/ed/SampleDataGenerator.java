package net.jacobpeterson.ed;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.concurrent.ThreadLocalRandom;

public class SampleDataGenerator {

    public static Pair<INDArray, INDArray> generate2x2Sample(SampleDirection direction) {
        SampleDirection[] sampleDirections = SampleDirection.values();
        ThreadLocalRandom random = ThreadLocalRandom.current();

        if (direction == null) {
            direction = sampleDirections[random.nextInt(sampleDirections.length)];
        }

        int imageLength = 2 * 2; // 2x2 grayscale image
        int numberOfDirections = sampleDirections.length;

        double[] featureArray = new double[imageLength]; // Feature
        double[] labelArray = new double[numberOfDirections]; // Label

        double tinyPixelOffset = 0.05;
        double similarPixelBase = random.nextDouble();

        switch (direction) {
            case VERTICAL:
                featureArray[0] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);
//                featureArray[1] = clamp(random.nextDouble() +
//                        random.nextDouble(-similarPixelBase / 3, similarPixelBase / 3), 0, 1);
                featureArray[1] = random.nextDouble(0.33D);
                featureArray[2] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);
//                featureArray[3] = clamp(random.nextDouble() +
//                        random.nextDouble(-similarPixelBase / 3, similarPixelBase / 3), 0, 1);
                featureArray[3] = random.nextDouble(0.66D, 1D);

                // Perform raster column switching for half of the samples
                if (random.nextDouble() > 0.5d) {
                    double topRightTemp = featureArray[1];
                    double bottomRightTemp = featureArray[3];

                    featureArray[1] = featureArray[0];
                    featureArray[3] = featureArray[2];

                    featureArray[0] = topRightTemp;
                    featureArray[2] = bottomRightTemp;
                }
                break;
            case HORIZONTAL:
                featureArray[0] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);
                featureArray[1] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);
//                featureArray[2] = clamp(random.nextDouble() +
//                        random.nextDouble(-similarPixelBase / 2, similarPixelBase / 2), 0, 1);
//                featureArray[3] = clamp(random.nextDouble() +
//                        random.nextDouble(-similarPixelBase / 2, similarPixelBase / 2), 0, 1);
                featureArray[2] = random.nextDouble(0.33D);
                featureArray[3] = random.nextDouble(0.66D, 1D);

                // Perform raster row switching for half of the samples
                if (random.nextDouble() > 0.5d) {
                    double topLeftTemp = featureArray[0];
                    double topRightTemp = featureArray[1];

                    featureArray[0] = featureArray[2];
                    featureArray[1] = featureArray[3];

                    featureArray[2] = topLeftTemp;
                    featureArray[3] = topRightTemp;
                }
                break;
            case DIAGONAL:
                featureArray[0] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);
//                featureArray[1] = clamp(random.nextDouble() +
//                        random.nextDouble(-similarPixelBase / 2, similarPixelBase / 2), 0, 1);
                featureArray[1] = random.nextDouble(0.33D);
//                featureArray[2] = clamp(random.nextDouble() +
//                        random.nextDouble(-similarPixelBase / 2, similarPixelBase / 2), 0, 1);
                featureArray[2] = random.nextDouble(0.66D, 1D);
                featureArray[3] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);

                // Perform raster diagonal switching for half of the samples
                if (random.nextDouble() > 0.5d) {
                    double topLeftTemp = featureArray[0];
                    double topRightTemp = featureArray[1];

                    featureArray[0] = featureArray[2];
                    featureArray[1] = featureArray[3];

                    featureArray[2] = topLeftTemp;
                    featureArray[3] = topRightTemp;
                }
                break;
            case SOLID:
                featureArray[0] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);
                featureArray[1] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);
                featureArray[2] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);
                featureArray[3] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);
                break;
            case UNKNOWN:
                featureArray[0] = random.nextDouble(0, 0.25);
                featureArray[1] = random.nextDouble(0.25, 0.5);
                featureArray[2] = random.nextDouble(0.5, 0.75);
                featureArray[3] = random.nextDouble(0.75, 1);
                shuffleArray(random, featureArray);
                break;
            default:
                throw new IllegalStateException("No case known for: " + direction);
        }

        // Assign which output neuron should be a 1 and which ones should be 0
        labelArray[direction.getOutputNeuronIndex()] = 1d;

        INDArray featureNDArray = Nd4j.create(featureArray);
        INDArray labelNDArray = Nd4j.create(labelArray);

        return new Pair<>(featureNDArray, labelNDArray);
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    // Implementing Fisher–Yates shuffle from StackOverflow
    private static void shuffleArray(ThreadLocalRandom random, double[] ar) {
        for (int i = ar.length - 1; i > 0; i--) {
            int index = random.nextInt(i + 1);
            // Simple swap
            double a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }

    public enum SampleDirection {
        // DON'T CHANGE THIS ORDER (unless you want to recreate the network)
        VERTICAL(0),
        HORIZONTAL(1),
        DIAGONAL(2),
        SOLID(3),
        UNKNOWN(4);

        private final int outputNeuronIndex;

        SampleDirection(int outputNeuronIndex) {
            this.outputNeuronIndex = outputNeuronIndex;
        }

        public int getOutputNeuronIndex() {
            return outputNeuronIndex;
        }

        public static SampleDirection fromNeuronIndex(int outputNeuronIndex) {
            for (SampleDirection sampleDirection : values()) {
                if (sampleDirection.getOutputNeuronIndex() == outputNeuronIndex) {
                    return sampleDirection;
                }
            }
            return null;
        }
    }
}
