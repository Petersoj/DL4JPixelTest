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
                featureArray[1] = random.nextDouble();
                featureArray[2] = clamp(similarPixelBase + random.nextDouble(-tinyPixelOffset, tinyPixelOffset), 0, 1);
                featureArray[3] = random.nextDouble();

                // Perform raster column switching for half of the samples
                if (random.nextDouble() > 0.5d) {
                    double topRightTemp = featureArray[1];
                    double bottomRightTemp = featureArray[3];

                }
                break;
            case HORIZONTAL:
                break;
            case DIAGONAL:
                break;
            case SOLID:
                break;
            case UNKNOWN:
                break;
            default:
                throw new IllegalStateException("No case known for: " + direction);
        }

        INDArray featureNDArray = Nd4j.create(featureArray);
        INDArray labelNDArray = Nd4j.create(labelArray);

        return new Pair<>(featureNDArray, labelNDArray);
    }

    public static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    public enum SampleDirection {
        // DON'T CHANGE THIS ORDER (unless you want to recreate the network)
        VERTICAL,
        HORIZONTAL,
        DIAGONAL,
        SOLID,
        UNKNOWN
    }
}
