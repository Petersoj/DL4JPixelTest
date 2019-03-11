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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DL4JTest {

    public static void main(String[] args) throws Exception {
        File networkSaveFile = new File(System.getProperty("user.home"), ".pixel_direction_nn");

//        createAndTrainPixelDirectionNetwork(networkSaveFile);
        loadAndDisplayPixelDirectionNetwork(networkSaveFile);
    }

    public static void createAndTrainPixelDirectionNetwork(File networkSaveFile) throws Exception {
        int trainSampleSize = 100_000;
        int testSampleSize = trainSampleSize;
        int inputNeurons = 2 * 2; // 2x2 grayscale image input
        int epochs = 20;
        double l2value = 1e-10;

        // Generate Training Data
        List<Pair<INDArray, INDArray>> trainList = new ArrayList<>();
        for (int i = 0; i < trainSampleSize; i++) {
            trainList.add(SampleDataGenerator.generate2x2Sample(null));
        }
        // Generate Testing Data
        List<Pair<INDArray, INDArray>> testList = new ArrayList<>();
        for (int i = 0; i < testSampleSize; i++) {
            testList.add(SampleDataGenerator.generate2x2Sample(null));
        }

        // Create DataSet iterators for generated samples
        INDArrayDataSetIterator dataSetIteratorTrain = new INDArrayDataSetIterator(trainList, 5);
        INDArrayDataSetIterator dataSetIteratorTest = new INDArrayDataSetIterator(testList, 5);

        // Configure Hyperparameters for NN
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(l2value)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(inputNeurons)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                ).layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
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
            private int printIterations = trainSampleSize / 10;

            @Override
            public void iterationDone(Model model, int iteration, int epoch) {
                if (iteration % printIterations == 0) {
//                    System.out.print("\r");
                    System.out.println("Iteration: " + iteration + " Epoch: " + epoch + " Score: " + model.score());
                }
            }
        });
//        network.addListeners(new TimeIterationListener(2));
//        network.addListeners(new ScoreIterationListener(10)); // THIS NOW WORKS WITH THE UPDATED POM.XML


        System.out.println("Training network");
        long currentTime = System.currentTimeMillis();
        // Train network
        network.fit(dataSetIteratorTrain, epochs); // TRAIN NETWORK WITH CERTAIN EPOCH COUNT
        System.out.println("Training took: " + ((double) (System.currentTimeMillis() - currentTime) / 1000 / 60) +
                " minutes");

        if (networkSaveFile != null) {
            if (networkSaveFile.exists()) {
                System.out.println("Network Save File already exists! Not saving network to file.");
            } else {
                System.out.println("Saving Network to: " + networkSaveFile.getAbsolutePath());
                network.save(networkSaveFile, true);
            }
        }

        System.out.println("Evaluating network");
        // Evaluate Network
        Evaluation eval = network.evaluate(dataSetIteratorTest);
        System.out.println(eval.stats());
    }

    public static void loadAndDisplayPixelDirectionNetwork(File networkSaveFile) throws Exception {
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(networkSaveFile, true);
        System.out.println("Loaded network from: " + networkSaveFile.getAbsolutePath());

        // Generate Testing Data
        int testSampleSize = 50_000;
        List<Pair<INDArray, INDArray>> testList = new ArrayList<>();
        for (int i = 0; i < testSampleSize; i++) {
            testList.add(SampleDataGenerator.generate2x2Sample(null));
        }

        // Create DataSet iterators for generated samples
        INDArrayDataSetIterator dataSetIteratorTest = new INDArrayDataSetIterator(testList, 1);

        System.out.println("Evaluating network");
        // Evaluate Network
        Evaluation eval = network.evaluate(dataSetIteratorTest);
        System.out.println(eval.stats());

        System.out.println("Executing intermittent testing and displaying");
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
                    "SAME?: " + (predictedDirection.equals(expectedDirection) ? "Yes " : "No ") +
                            "Expected: " + expectedDirection.name() +
                            " Prediction: " + predictedDirection.name()
            );
            Thread.sleep(3000);
        }
    }

    public static void displayAndTestPixelDirections() throws Exception {
        PixelDisplay pixelDisplay2 = new PixelDisplay();
        while (true) {
            Pair<INDArray, INDArray> testRaster =
                    SampleDataGenerator.generate2x2Sample(null);
            pixelDisplay2.setGrayScaleRaster(testRaster.getKey());

            for (int i = 0; i < 5; i++) {
                if (testRaster.getValue().getDouble(i) == 1D) {
                    System.out.println(SampleDataGenerator.SampleDirection.values()[i].name());
                    break;
                }
            }

            Thread.sleep(2000);
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
