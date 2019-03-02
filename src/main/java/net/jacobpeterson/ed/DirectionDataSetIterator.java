package net.jacobpeterson.ed;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.primitives.Pair;

import java.util.function.Consumer;

public class DirectionDataSetIterator extends INDArrayDataSetIterator {

    public DirectionDataSetIterator(Iterable<Pair<INDArray, INDArray>> iterable, int batchSize) {
        super(iterable, batchSize);
    }

    @Override
    public void forEachRemaining(Consumer<? super DataSet> action) {
        super.forEachRemaining(action);
    }
}
