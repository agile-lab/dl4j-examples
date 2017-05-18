package org.deeplearning4j.examples.deeplogo;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Lists;
import lombok.Getter;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by paolo on 18/05/2017.
 */
public class MyMultiEpochIterator implements DataSetIterator{
    @VisibleForTesting
    protected int epochs = 0;
    protected int numEpochs;
    protected int batch = 0;
    protected int lastBatch = batch;
    protected ArrayList<DataSetIterator> iters;
    protected DataSet ds;
    protected List<DataSet> batchedDS = Lists.newArrayList();
    protected static final Logger log = LoggerFactory.getLogger(MultipleEpochsIterator.class);
    @Getter
    protected DataSetPreProcessor preProcessor;
    protected boolean newEpoch = false;
    protected int queueSize = 1;
    protected boolean async = false;
    protected AtomicLong iterationsCounter = new AtomicLong(0);
    protected long totalIterations = Long.MAX_VALUE;

    protected int currentIter = 0;
    protected int itersSize;

    public MyMultiEpochIterator(int numEpochs, Collection<DataSetIterator> iters) {
        this.numEpochs = numEpochs;
        this.iters =  new ArrayList(iters);
        itersSize = iters.size();
    }




    /**
     * Like the standard next method but allows a
     * customizable number of examples returned
     *
     * @param num the number of examples
     * @return the next data applyTransformToDestination
     */
    @Override
    public DataSet next(int num) {
        DataSet next;
        batch++;
        iterationsCounter.incrementAndGet();
        if (iters == null) {
            // return full DataSet

            next = ds;
            if (epochs < numEpochs)
                trackEpochs();

        } else {

            DataSetIterator iter = iters.get(currentIter%itersSize);
            currentIter++;

            next = iter.next();
            if (!iter.hasNext()) {
                trackEpochs();
                // track number of epochs and won't reset if it's over
                if (epochs < numEpochs) {
                    for(DataSetIterator iteri : iters){
                        iteri.reset();
                    }

                    lastBatch = batch;
                    batch = 0;
                }
            }
        }
        if (preProcessor != null)
            preProcessor.preProcess(next);
        return next;
    }

    @Override
    public DataSet next() {
        return next(-1);
    }

    public void trackEpochs() {
        epochs++;
        newEpoch = true;
    }



    /**
     * Total examples in the iterator
     *
     * @return
     */
    @Override
    public int totalExamples() {
        return iters.get(0).totalExamples()*itersSize;
    }

    /**
     * Input columns for the dataset
     *
     * @return
     */
    @Override
    public int inputColumns() {
        return iters.get(0).inputColumns();
    }

    /**
     * The number of labels for the dataset
     *
     * @return
     */
    @Override
    public int totalOutcomes() {
        return iters.get(0).totalOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return iters.get(0).resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return !async;
    }

    /**
     * Resets the iterator back to the beginning
     */
    @Override
    public void reset() {
        epochs = 0;
        lastBatch = batch;
        batch = 0;
        iterationsCounter.set(0);
        currentIter = 0;
        for(DataSetIterator iteri : iters){
            iteri.reset();
        }
    }

    /**
     * Batch size
     *
     * @return
     */
    @Override
    public int batch() {
        return iters.get(0).batch();
    }

    /**
     * The current cursor if applicable
     *
     * @return
     */
    @Override
    public int cursor() {
        return iters.get(0).cursor();
    }

    /**
     * Total number of examples in the dataset
     *
     * @return
     */
    @Override
    public int numExamples() {
        return iters.get(0).numExamples()*itersSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = (DataSetPreProcessor) preProcessor;
    }


    @Override
    public List<String> getLabels() {
        return iters.get(0).getLabels();
    }


    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        if (iterationsCounter.get() >= totalIterations)
            return false;

        if (newEpoch) {
            log.info("Epoch " + epochs + ", number of batches completed " + lastBatch);
            newEpoch = false;
        }
        if (iters == null)
            return (epochs < numEpochs) && ((!batchedDS.isEmpty() && batchedDS.size() > batch) || batchedDS.isEmpty());
        else
            // either there are still epochs to complete or its the first epoch
            return (epochs < numEpochs) || (iters.get(currentIter%itersSize).hasNext() && (epochs == 0 || epochs == numEpochs));
    }

    /**
     * Removes from the underlying collection the last element returned
     * by this iterator (optional operation).  This method can be called
     * only once per call to {@link #next}.  The behavior of an iterator
     * is unspecified if the underlying collection is modified while the
     * iteration is in progress in any way other than by calling this
     * method.
     *
     * @throws UnsupportedOperationException if the {@code remove}
     *                                       operation is not supported by this iterator
     * @throws IllegalStateException         if the {@code next} method has not
     *                                       yet been called, or the {@code remove} method has already
     *                                       been called after the last call to the {@code next}
     *                                       method
     */
    @Override
    public void remove() {

        for(DataSetIterator iteri : iters){
            iteri.remove();
        }
    }

}
