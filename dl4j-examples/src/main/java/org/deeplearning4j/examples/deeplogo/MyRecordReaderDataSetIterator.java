//package org.deeplearning4j.examples.deeplogo;
//
//import annotation.MultiNDArrayWritable;
//import lombok.Getter;
//import lombok.Setter;
//import org.datavec.api.io.WritableConverter;
//import org.datavec.api.io.converters.SelfWritableConverter;
//import org.datavec.api.io.converters.WritableConverterException;
//import org.datavec.api.records.Record;
//import org.datavec.api.records.metadata.RecordMetaData;
//import org.datavec.api.records.reader.RecordReader;
//import org.datavec.api.records.reader.SequenceRecordReader;
//import org.datavec.api.writable.Writable;
//import org.datavec.common.data.NDArrayWritable;
//import org.deeplearning4j.exception.DL4JInvalidInputException;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.dataset.DataSet;
//import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
//import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
//import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.util.FeatureUtil;
//
//import java.io.IOException;
//import java.util.ArrayList;
//import java.util.Collections;
//import java.util.Iterator;
//import java.util.List;
//
///**
// * Created by andlatel on 24/05/2017.
// */
//public class MyRecordReaderDataSetIterator implements DataSetIterator {
//    protected RecordReader recordReader;
//    protected WritableConverter converter;
//    protected int batchSize = 10;
//    protected int maxNumBatches = -1;
//    protected int batchNum = 0;
//    protected int labelIndex = -1;
//    protected int labelIndexTo = -1;
//    protected int numPossibleLabels = -1;
//    protected Iterator<List<Writable>> sequenceIter;
//    protected DataSet last;
//    protected boolean useCurrent = false;
//    protected boolean regression = false;
//    @Getter
//    protected DataSetPreProcessor preProcessor;
//
//    @Getter
//    @Setter
//    private boolean collectMetaData = false;
//
//    public MyRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize) {
//        this(recordReader, converter, batchSize, -1,
//            recordReader.getLabels() == null ? -1 : recordReader.getLabels().size());
//    }
//
//    public MyRecordReaderDataSetIterator(RecordReader recordReader, int batchSize) {
//        this(recordReader, new SelfWritableConverter(), batchSize, -1,
//            recordReader.getLabels() == null ? -1 : recordReader.getLabels().size());
//    }
//
//    /**
//     * Main constructor for classification. This will convert the input class index (at position labelIndex, with integer
//     * values 0 to numPossibleLabels-1 inclusive) to the appropriate one-hot output/labels representation.
//     *
//     * @param recordReader         RecordReader: provides the source of the data
//     * @param batchSize            Batch size (number of examples) for the output DataSet objects
//     * @param labelIndex           Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
//     * @param numPossibleLabels    Number of classes (possible labels) for classification
//     */
//    public MyRecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndex,
//                                       int numPossibleLabels) {
//        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels);
//    }
//
//    public MyRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
//                                       int labelIndex, int numPossibleLabels, boolean regression) {
//        this(recordReader, converter, batchSize, labelIndex, numPossibleLabels, -1, regression);
//    }
//
//    public MyRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
//                                       int labelIndex, int numPossibleLabels) {
//        this(recordReader, converter, batchSize, labelIndex, numPossibleLabels, -1, false);
//    }
//
//    public MyRecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels,
//                                       int maxNumBatches) {
//        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels, maxNumBatches, false);
//    }
//
//    /**
//     * Main constructor for multi-label regression (i.e., regression with multiple outputs)
//     *
//     * @param recordReader      RecordReader to get data from
//     * @param labelIndexFrom    Index of the first regression target
//     * @param labelIndexTo      Index of the last regression target, inclusive
//     * @param batchSize         Minibatch size
//     * @param regression        Require regression = true. Mainly included to avoid clashing with other constructors previously defined :/
//     */
//    public MyRecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndexFrom, int labelIndexTo,
//                                       boolean regression) {
//        this(recordReader, new SelfWritableConverter(), batchSize, labelIndexFrom, labelIndexTo, -1, -1, regression);
//    }
//
//
//    public MyRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
//                                       int labelIndex, int numPossibleLabels, int maxNumBatches, boolean regression) {
//        this(recordReader, converter, batchSize, labelIndex, labelIndex, numPossibleLabels, maxNumBatches, regression);
//    }
//
//
//    /**
//     * Main constructor
//     *
//     * @param recordReader      the recordreader to use
//     * @param converter         the batch size
//     * @param maxNumBatches     Maximum number of batches to return
//     * @param labelIndexFrom    the index of the label (for classification), or the first index of the labels for multi-output regression
//     * @param labelIndexTo      only used if regression == true. The last index _inclusive_ of the multi-output regression
//     * @param numPossibleLabels the number of possible labels for classification. Not used if regression == true
//     * @param regression        if true: regression. If false: classification (assume labelIndexFrom is a
//     */
//    public MyRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
//                                       int labelIndexFrom, int labelIndexTo, int numPossibleLabels, int maxNumBatches,
//                                       boolean regression) {
//        this.recordReader = recordReader;
//        this.converter = converter;
//        this.batchSize = batchSize;
//        this.maxNumBatches = maxNumBatches;
//        this.labelIndex = labelIndexFrom;
//        this.labelIndexTo = labelIndexTo;
//        this.numPossibleLabels = numPossibleLabels;
//        this.regression = regression;
//    }
//
//
//    @Override
//    public DataSet next(int num) {
//        if (useCurrent) {
//            useCurrent = false;
//            if (preProcessor != null)
//                preProcessor.preProcess(last);
//            return last;
//        }
//
//        List<DataSet> dataSets = new ArrayList<>();
//        List<RecordMetaData> meta = (collectMetaData ? new ArrayList<RecordMetaData>() : null);
//        for (int i = 0; i < num; i++) {
//            if (!hasNext())
//                break;
//            if (recordReader instanceof SequenceRecordReader) {
//                if (sequenceIter == null || !sequenceIter.hasNext()) {
//                    List<List<Writable>> sequenceRecord = ((SequenceRecordReader) recordReader).sequenceRecord();
//                    sequenceIter = sequenceRecord.iterator();
//                }
//
//                List<Writable> record = sequenceIter.next();
//                dataSets.add(getDataSet(record));
//            } else {
//                if (collectMetaData) {
//                    Record record = recordReader.nextRecord();
//                    dataSets.add(getDataSet(record.getRecord()));
//                    meta.add(record.getMetaData());
//                } else {
//                    List<Writable> record = recordReader.next();
//                    dataSets.add(myGetDataSet(record));
//
//                    //batchNum++;
//                    //return (myGetDataSet(record));
//                }
//            }
//        }
//        batchNum++;
//
//        if (dataSets.isEmpty())
//            return new DataSet();
//
//        DataSet ret = MyDataSet.merge(dataSets);
//
//        if (collectMetaData) {
//            ret.setExampleMetaData(meta);
//        }
//        last = ret;
//        if (preProcessor != null)
//            preProcessor.preProcess(ret);
//        //Add label name values to dataset
//        if (recordReader.getLabels() != null)
//            ret.setLabelNames(recordReader.getLabels());
//        return ret;
//    }
//
//
//    private DataSet myGetDataSet(List<Writable> record) {
//
//        List<Writable> regionsImage;
//        List<Writable> singleRegion;
//        MultiNDArrayWritable regionsImageWritable = (MultiNDArrayWritable) record.get(0);
//        regionsImage = scala.collection.JavaConversions.seqAsJavaList(regionsImageWritable.list());
//
//
//        //allow people to specify label index as -1 and infer the last possible label
//        if (numPossibleLabels >= 1 && labelIndex < 0) {
//            labelIndex = record.size() - 1;
//        }
//
//        INDArray label = null;
//        INDArray featureVector = null;
//        int featureCount = 0;
//        int labelCount = 0;
//
//        /*currList.stream().map(l->{
//            return 1;
//        }).collect(Collectors.toList());*/
//
//        DataSet finalDataset = new org.nd4j.linalg.dataset.DataSet();
//        List<DataSet> dl = new ArrayList<DataSet>();
//        Iterator<Writable> currListIter = regionsImage.iterator();
//        int i=0;
//        while(currListIter.hasNext()){
//            Writable currElem = currListIter.next();
//            if ( currElem instanceof NDArrayWritable) {
//                if (!regression) {
//                    label = FeatureUtil.toOutcomeVector((int) Double.parseDouble(record.get(1).toString()), numPossibleLabels);
//                }
//
//                NDArrayWritable ndArrayWritable = (NDArrayWritable) currElem;
//                featureVector = ndArrayWritable.get();
//                //DataSet dat = new DataSet(featureVector, label);
//                //finalDataset.addRow(new DataSet(featureVector, label), i);
//                dl.add(new DataSet(featureVector, label));
//                i++;
//
//            }
//        }
//
//        MyDataSet res = new MyDataSet(dl);
//        return res;
//    }
//
//    private DataSet getDataSet(List<Writable> record) {
//        List<Writable> currList;
//        if (record instanceof List)
//            currList = record;
//        else
//            currList = new ArrayList<>(record);
//
//        //allow people to specify label index as -1 and infer the last possible label
//        if (numPossibleLabels >= 1 && labelIndex < 0) {
//            labelIndex = record.size() - 1;
//        }
//
//        INDArray label = null;
//        INDArray featureVector = null;
//        int featureCount = 0;
//        int labelCount = 0;
//
//        //no labels
//        if (currList.size() == 2 && currList.get(1) instanceof NDArrayWritable
//            && currList.get(0) instanceof NDArrayWritable && currList.get(0) == currList.get(1)) {
//            NDArrayWritable writable = (NDArrayWritable) currList.get(0);
//            return new DataSet(writable.get(), writable.get());
//        }
//        if (currList.size() == 2 && currList.get(0) instanceof NDArrayWritable) {
//            if (!regression) {
//                label = FeatureUtil.toOutcomeVector((int) Double.parseDouble(currList.get(1).toString()),
//                    numPossibleLabels);
//            } else {
//                if (currList.get(1) instanceof NDArrayWritable) {
//                    label = ((NDArrayWritable) currList.get(1)).get();
//                } else {
//                    label = Nd4j.scalar(currList.get(1).toDouble());
//                }
//            }
//            NDArrayWritable ndArrayWritable = (NDArrayWritable) currList.get(0);
//            featureVector = ndArrayWritable.get();
//            return new DataSet(featureVector, label);
//        }
//
//        for (int j = 0; j < currList.size(); j++) {
//            Writable current = currList.get(j);
//            //ndarray writable is an insane slow down herecd
//            if (!(current instanceof NDArrayWritable) && current.toString().isEmpty())
//                continue;
//
//            if (regression && j == labelIndex && j == labelIndexTo && current instanceof NDArrayWritable) {
//                //Case: NDArrayWritable for the labels
//                label = ((NDArrayWritable) current).get();
//            } else if (regression && j >= labelIndex && j <= labelIndexTo) {
//                //This is the multi-label regression case
//                if (label == null)
//                    label = Nd4j.create(1, (labelIndexTo - labelIndex + 1));
//                label.putScalar(labelCount++, current.toDouble());
//            } else if (labelIndex >= 0 && j == labelIndex) {
//                //single label case (classification, etc)
//                if (converter != null)
//                    try {
//                        current = converter.convert(current);
//                    } catch (WritableConverterException e) {
//                        e.printStackTrace();
//                    }
//                if (numPossibleLabels < 1)
//                    throw new IllegalStateException("Number of possible labels invalid, must be >= 1");
//                if (regression) {
//                    label = Nd4j.scalar(current.toDouble());
//                } else {
//                    int curr = current.toInt();
//                    if (curr < 0 || curr >= numPossibleLabels) {
//                        throw new DL4JInvalidInputException(
//                            "Invalid classification data: expect label value (at label index column = "
//                                + labelIndex + ") to be in range 0 to "
//                                + (numPossibleLabels - 1)
//                                + " inclusive (0 to numClasses-1, with numClasses="
//                                + numPossibleLabels + "); got label value of " + current);
//                    }
//                    label = FeatureUtil.toOutcomeVector(curr, numPossibleLabels);
//                }
//            } else {
//                try {
//                    double value = current.toDouble();
//                    if (featureVector == null) {
//                        if (regression && labelIndex >= 0) {
//                            //Handle the possibly multi-label regression case here:
//                            int nLabels = labelIndexTo - labelIndex + 1;
//                            featureVector = Nd4j.create(1, currList.size() - nLabels);
//                        } else {
//                            //Classification case, and also no-labels case
//                            featureVector = Nd4j.create(labelIndex >= 0 ? currList.size() - 1 : currList.size());
//                        }
//                    }
//                    featureVector.putScalar(featureCount++, value);
//                } catch (UnsupportedOperationException e) {
//                    // This isn't a scalar, so check if we got an array already
//                    if (current instanceof NDArrayWritable) {
//                        assert featureVector == null;
//                        featureVector = ((NDArrayWritable) current).get();
//                    } else {
//                        throw e;
//                    }
//                }
//            }
//        }
//
//        return new DataSet(featureVector, labelIndex >= 0 ? label : featureVector);
//    }
//
//    @Override
//    public int totalExamples() {
//        throw new UnsupportedOperationException();
//    }
//
//    @Override
//    public int inputColumns() {
//        if (last == null) {
//            DataSet next = next();
//            last = next;
//            useCurrent = true;
//            return next.numInputs();
//        } else
//            return last.numInputs();
//
//    }
//
//    @Override
//    public int totalOutcomes() {
//        if (last == null) {
//            DataSet next = next();
//            last = next;
//            useCurrent = true;
//            return next.numOutcomes();
//        } else
//            return last.numOutcomes();
//
//
//    }
//
//    @Override
//    public boolean resetSupported() {
//        return true;
//    }
//
//    @Override
//    public boolean asyncSupported() {
//        return true;
//    }
//
//    @Override
//    public void reset() {
//        batchNum = 0;
//        recordReader.reset();
//    }
//
//    @Override
//    public int batch() {
//        return batchSize;
//    }
//
//    @Override
//    public int cursor() {
//        throw new UnsupportedOperationException();
//
//    }
//
//    @Override
//    public int numExamples() {
//        throw new UnsupportedOperationException();
//    }
//
//    @Override
//    public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
//        this.preProcessor = preProcessor;
//    }
//
//    @Override
//    public DataSetPreProcessor getPreProcessor() {
//        return null;
//    }
//
//    @Override
//    public boolean hasNext() {
//        return (recordReader.hasNext() && (maxNumBatches < 0 || batchNum < maxNumBatches));
//    }
//
//    @Override
//    public DataSet next() {
//        return next(batchSize);
//    }
//
//    @Override
//    public void remove() {
//        throw new UnsupportedOperationException();
//    }
//
//    @Override
//    public List<String> getLabels() {
//        return recordReader.getLabels();
//    }
//
//    /**
//     * Load a single example to a DataSet, using the provided RecordMetaData.
//     * Note that it is more efficient to load multiple instances at once, using {@link #loadFromMetaData(List)}
//     *
//     * @param recordMetaData RecordMetaData to load from. Should have been produced by the given record reader
//     * @return DataSet with the specified example
//     * @throws IOException If an error occurs during loading of the data
//     */
//    public DataSet loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
//        return loadFromMetaData(Collections.singletonList(recordMetaData));
//    }
//
//    /**
//     * Load a multiple examples to a DataSet, using the provided RecordMetaData instances.
//     *
//     * @param list List of RecordMetaData instances to load from. Should have been produced by the record reader provided
//     *             to the RecordReaderDataSetIterator constructor
//     * @return DataSet with the specified examples
//     * @throws IOException If an error occurs during loading of the data
//     */
//    public DataSet loadFromMetaData(List<RecordMetaData> list) throws IOException {
//        List<Record> records = recordReader.loadFromMetaData(list);
//        List<DataSet> dataSets = new ArrayList<>();
//        List<RecordMetaData> meta = new ArrayList<>();
//        for (Record r : records) {
//            dataSets.add(getDataSet(r.getRecord()));
//            meta.add(r.getMetaData());
//        }
//
//        if (dataSets.isEmpty()) {
//            return new DataSet();
//        }
//
//        DataSet ret = DataSet.merge(dataSets);
//        ret.setExampleMetaData(meta);
//        last = ret;
//        if (preProcessor != null)
//            preProcessor.preProcess(ret);
//        if (recordReader.getLabels() != null)
//            ret.setLabelNames(recordReader.getLabels());
//        return ret;
//    }
//}
//
