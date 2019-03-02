package net.jacobpeterson.ed;

import com.sun.tools.javac.util.List;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import java.lang.Iterable;

import java.io.IOException;
import java.util.Iterator;

public class DL4JTest {

    public static void main(String[] args) throws IOException {
        executeMNISTExample();
        executeDirectionClassifier();
    }

    public static void executeDirectionClassifier() {
//        Iterator<Pair<INDArray, INDArray>> it =
//                List.of(SampleDataGenerator.generate2x2Sample(SampleDataGenerator.SampleDirection.DIAGONAL)).iterator();
//        new INDArrayDataSetIterator(it, 2);
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
        network.setListeners(new ScoreIterationListener(140));
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
