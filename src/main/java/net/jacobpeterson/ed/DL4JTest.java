package net.jacobpeterson.ed;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class DL4JTest {

    public static void main(String[] args) throws Exception {
//        executeMNISTExample();
        executeDirectionClassifier();
    }

    public static void executeDirectionClassifier() throws Exception {

//        PixelDisplay pixelDisplay2 = new PixelDisplay();
//        int i2 = 6;
//        while (i2 > 4) {
//            Pair<INDArray, INDArray> testRaster =
//                    SampleDataGenerator.generate2x2Sample(SampleDataGenerator.SampleDirection.UNKNOWN);
//            pixelDisplay2.setGrayScaleRaster(testRaster.getKey());
//            Thread.sleep(2000);
//            for (int i = 0; i < 5; i++) {
//                if (testRaster.getValue().getDouble(i) == 1D) {
//                    System.out.println(SampleDataGenerator.SampleDirection.values()[i].name());
//                    break;
//                }
//            }
//        }

        System.out.println("Generating Samples");

        int batchSize = 50_000;

        // Generate Training Data
        List<Pair<INDArray, INDArray>> trainList = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            trainList.add(SampleDataGenerator.generate2x2Sample(null));
        }
        INDArrayDataSetIterator dataSetIteratorTrain = new INDArrayDataSetIterator(trainList, 1);

        // Generate Testing Data
        List<Pair<INDArray, INDArray>> testList = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            testList.add(SampleDataGenerator.generate2x2Sample(null));
        }
        INDArrayDataSetIterator dataSetIteratorTest = new INDArrayDataSetIterator(testList, 1);

        // Configure Hyperparameters
        int imageLength = 2 * 2;
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(1e-8)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(imageLength)
                        .nOut(4)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                ).layer(new DenseLayer.Builder()
                        .nIn(4)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                ).layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(5)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                ).layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(5)
                        .nOut(SampleDataGenerator.SampleDirection.values().length)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backpropType(BackpropType.Standard)
                .pretrain(false)
                .build();

        // Construct Network
        System.out.println("Constructing Network");
        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();
        network.addListeners(new BaseTrainingListener() {
            private int printIterations = batchSize / 10;

            @Override
            public void iterationDone(Model model, int iteration, int epoch) {
                if (iteration % printIterations == 0) {
                    System.out.println("Iteration: " + iteration + " Epoch: " + epoch + " Score: " + model.score());
                }
            }
        });
//        network.addListeners(new ScoreIterationListener(10)); // THIS NOW WORKS WITH THE UPDATED POM.XML
        network.printConfiguration();

        // Train network
        System.out.println("Training network");
        long currentTime = System.currentTimeMillis();
        network.fit(dataSetIteratorTrain, 5); // TRAIN NETWORK
        System.out.println("Training took: " +
                TimeUnit.MILLISECONDS.convert((System.currentTimeMillis() - currentTime), TimeUnit.SECONDS) + " " +
                "seconds");

        System.out.println("Evaluating network");
        Evaluation eval = network.evaluate(dataSetIteratorTest);
        System.out.println(eval.stats());

        // Test and display pixel
        PixelDisplay pixelDisplay = new PixelDisplay();
        while (true) {
            Pair<INDArray, INDArray> testRaster = SampleDataGenerator.generate2x2Sample(null);
            pixelDisplay.setGrayScaleRaster(testRaster.getKey());

            INDArray expectedNDArray = testRaster.getValue();
            SampleDataGenerator.SampleDirection expectedDirection = null;
            for (int i = 0; i < expectedNDArray.length(); i++) {
                if (expectedNDArray.getDouble(i) == 1D) {
                    expectedDirection = SampleDataGenerator.SampleDirection.fromNeuronIndex(i);
                    break;
                }
            }
            SampleDataGenerator.SampleDirection predictedDirection =
                    SampleDataGenerator.SampleDirection.fromNeuronIndex(network.predict(testRaster.getKey())[0]);
            System.out.println(
                    "WORKED?: " + (predictedDirection.equals(expectedDirection) ? "Yes " : "No ") +
                            "Expected: " + expectedDirection.name() +
                            " Prediction: " + predictedDirection.name()
            );
            Thread.sleep(2500);
        }
    }

    public static void executeMNISTExample() throws IOException {
        System.out.println("Loading EmnistDataSet");
        System.out.println("HI:" + DL4JResources.getDirectory(ResourceType.DATASET, "EMNIST").getAbsolutePath());

        EmnistDataSetIterator.Set emnistSet = EmnistDataSetIterator.Set.DIGITS;

        EmnistDataSetIterator emnistTrain = new EmnistDataSetIterator(emnistSet, 15, true);
        EmnistDataSetIterator emnistTest = new EmnistDataSetIterator(emnistSet, 15, false);

        int outputNum = EmnistDataSetIterator.numLabels(emnistSet);
        int rngSeed = 123;
        int rows = 28;
        int cols = 28;

        System.out.println("Configuration");

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(rows * cols)
                        .nOut(120)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                ).layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(120)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build();

        System.out.println("Init network");

        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.setListeners(new ScoreIterationListener(1));
        network.init();

        System.out.println("Configuration");

        System.out.println("Training network");
        network.fit(emnistTrain);

        System.out.println("Evaluating");
        Evaluation eval = network.evaluate(emnistTest);
        System.out.println(eval.stats());

//        List<String> strings = network.predict(emnistTest.next().get(0));
//        System.out.println(strings.toString());
    }
}
