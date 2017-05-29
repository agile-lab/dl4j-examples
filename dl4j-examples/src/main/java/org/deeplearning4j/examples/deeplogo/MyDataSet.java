package org.deeplearning4j.examples.deeplogo;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * Created by andlatel on 25/05/2017.
 */
public class MyDataSet extends DataSet {

    List<DataSet> listDataSet;
    public MyDataSet() {
        listDataSet= new ArrayList<DataSet>();
    }

    public MyDataSet( List<DataSet> listDataSet) {
        this.listDataSet= listDataSet;
    }

    public List<DataSet> getListDataSet() {
        return listDataSet;
    }

    public void setListDataSet(List<DataSet> listDataSet) {
        this.listDataSet = listDataSet;
    }

    /**
     * Merge the list of datasets in to one list.
     * All the rows are merged in to one dataset
     *
     * @param data the data to merge
     * @return a single dataset
     */
    public static DataSet merge(List<DataSet> data) {

        // contains all region of one image
        INDArray featuresOutForImage = null;
        INDArray labelsOutForImage = null;
        INDArray featuresMaskOutForImage = null;
        INDArray labelsMaskOutForImage = null;

        INDArray[] featuresImageToMerge = new INDArray[data.size()];
        INDArray[] labelsImageToMerge = new INDArray[data.size()];

        if (data.isEmpty())
            throw new IllegalArgumentException("Unable to merge empty dataset");

        int countImage = 0;

        //iterate on image
        for(DataSet image: data){
            //retrieve all region for image
            List<DataSet> regionList = ((MyDataSet) image).getListDataSet();

            INDArray[] featuresRegionSameImageToMerge = new INDArray[regionList.size()];
            INDArray[] labelsSameImageToMerge = new INDArray[regionList.size()];

            int countRegionForImage = 0;
            boolean hasFeaturesMaskArray = false;
            boolean hasLabelsMaskArray = false;

            int rankFeatures = 0;
            int rankLabels = 0;

            //contains region list for each global image
            for(DataSet region: regionList) {
                rankFeatures = region.getFeatures().rank();
                rankLabels = region.getLabels().rank();

                featuresRegionSameImageToMerge[countRegionForImage] = region.getFeatureMatrix();
                labelsSameImageToMerge[countRegionForImage++] = region.getLabels();

                if (rankFeatures == 3 || rankLabels == 3) {
                    hasFeaturesMaskArray = hasFeaturesMaskArray | (region.getFeaturesMaskArray() != null);
                    hasLabelsMaskArray = hasLabelsMaskArray | (region.getLabelsMaskArray() != null);
                }

            }

            switch (rankFeatures) {
                case 2:
                    featuresOutForImage =  merge2d(featuresRegionSameImageToMerge);
                    featuresMaskOutForImage = null;
                    break;
                case 3:
                    //Time series data: may also have mask arrays...
                    INDArray[] featuresMasks = null;
                    if (hasFeaturesMaskArray) {
                        featuresMasks = new INDArray[featuresRegionSameImageToMerge.length];
                        countImage = 0;
                        for (DataSet ds : data) {
                            featuresMasks[countImage++] = ds.getFeaturesMaskArray();
                        }
                    }
                    INDArray[] temp = mergeTimeSeries(featuresRegionSameImageToMerge, featuresMasks);
                    featuresOutForImage =  temp[0];
                    featuresMaskOutForImage =  temp[1];
                    break;
                case 4:
                    featuresOutForImage = merge4dCnnData(featuresRegionSameImageToMerge);
                    featuresMaskOutForImage = null;
                    featuresImageToMerge[countImage] = featuresOutForImage;
                    break;
                default:
                    throw new IllegalStateException(
                        "Cannot merge examples: features rank must be in range 2 to 4 inclusive. First example features shape: "
                            + Arrays.toString(data.get(0).getFeatureMatrix().shape()));
            }

            switch (rankLabels) {
                case 2:
                    labelsOutForImage =  merge2d(labelsSameImageToMerge);
                    labelsMaskOutForImage = null;
                    labelsImageToMerge[countImage] = labelsOutForImage;
                    break;
                case 3:
                    //Time series data: may also have mask arrays...
                    INDArray[] labelsMasks = null;
                    if (hasLabelsMaskArray) {
                        labelsMasks = new INDArray[labelsSameImageToMerge.length];
                        countImage = 0;
                        for (DataSet ds : data) {
                            labelsMasks[countImage++] = ds.getLabelsMaskArray();
                        }
                    }
                    INDArray[] temp = mergeTimeSeries(labelsSameImageToMerge, labelsMasks);
                    labelsOutForImage =  temp[0];
                    labelsMaskOutForImage =  temp[1];

                    break;
                case 4:
                    labelsOutForImage =  merge4dCnnData(labelsSameImageToMerge);
                    labelsMaskOutForImage = null;
                    break;
                default:
                    throw new IllegalStateException(
                        "Cannot merge examples: labels rank must be in range 2 to 4 inclusive. First example labels shape: "
                            + Arrays.toString(data.get(0).getLabels().shape()));
            }
            countImage++;
        }

        //Merge Image Feature in new dimension
        INDArray mergeFetures5Dimension =  merge5dCnnData(featuresImageToMerge);
        //Merge Image Labels in new dimension
        INDArray mergeLabel =  mergeWithAdd3d(labelsImageToMerge);

        labelsMaskOutForImage = null;
        featuresMaskOutForImage = null;

        DataSet dataset = new DataSet(mergeFetures5Dimension, mergeLabel, featuresMaskOutForImage, labelsMaskOutForImage);
        return dataset;

        /*
        if (data.isEmpty())
            throw new IllegalArgumentException("Unable to merge empty dataset");

        DataSet first = data.get(0);

        int rankFeatures = first.getFeatures().rank();
        int rankLabels = first.getLabels().rank();

        INDArray[] featuresToMerge = new INDArray[data.size()];
        INDArray[] labelsToMerge = new INDArray[data.size()];
        int count = 0;
        boolean hasFeaturesMaskArray = false;
        boolean hasLabelsMaskArray = false;
        for (DataSet ds : data) {
            featuresToMerge[count] = ds.getFeatureMatrix();
            labelsToMerge[count++] = ds.getLabels();
            if (rankFeatures == 3 || rankLabels == 3) {
                hasFeaturesMaskArray = hasFeaturesMaskArray | (ds.getFeaturesMaskArray() != null);
                hasLabelsMaskArray = hasLabelsMaskArray | (ds.getLabelsMaskArray() != null);
            }
        }

        INDArray featuresOut;
        INDArray labelsOut;
        INDArray featuresMaskOut;
        INDArray labelsMaskOut;

        switch (rankFeatures) {
            case 2:
                featuresOut = merge2d(featuresToMerge);
                featuresMaskOut = null;
                break;
            case 3:
                //Time series data: may also have mask arrays...
                INDArray[] featuresMasks = null;
                if (hasFeaturesMaskArray) {
                    featuresMasks = new INDArray[featuresToMerge.length];
                    count = 0;
                    for (DataSet ds : data) {
                        featuresMasks[count++] = ds.getFeaturesMaskArray();
                    }
                }
                INDArray[] temp = mergeTimeSeries(featuresToMerge, featuresMasks);
                featuresOut = temp[0];
                featuresMaskOut = temp[1];
                break;
            case 4:
                featuresOut = merge4dCnnData(featuresToMerge);
                featuresMaskOut = null;
                break;
            default:
                throw new IllegalStateException(
                    "Cannot merge examples: features rank must be in range 2 to 4 inclusive. First example features shape: "
                        + Arrays.toString(data.get(0).getFeatureMatrix().shape()));
        }

        switch (rankLabels) {
            case 2:
                labelsOut = merge2d(labelsToMerge);
                labelsMaskOut = null;
                break;
            case 3:
                //Time series data: may also have mask arrays...
                INDArray[] labelsMasks = null;
                if (hasLabelsMaskArray) {
                    labelsMasks = new INDArray[labelsToMerge.length];
                    count = 0;
                    for (DataSet ds : data) {
                        labelsMasks[count++] = ds.getLabelsMaskArray();
                    }
                }
                INDArray[] temp = mergeTimeSeries(labelsToMerge, labelsMasks);
                labelsOut = temp[0];
                labelsMaskOut = temp[1];

                break;
            case 4:
                labelsOut = merge4dCnnData(labelsToMerge);
                labelsMaskOut = null;
                break;
            default:
                throw new IllegalStateException(
                    "Cannot merge examples: labels rank must be in range 2 to 4 inclusive. First example labels shape: "
                        + Arrays.toString(data.get(0).getLabels().shape()));
        }*/

        /*DataSet dataset = new DataSet(featuresOut, labelsOut, featuresMaskOut, labelsMaskOut);

        List<Serializable> meta = null;
        for (DataSet ds : data) {
            if (ds.getExampleMetaData() == null || ds.getExampleMetaData().size() != ds.numExamples()) {
                meta = null;
                break;
            }
            if (meta == null)
                meta = new ArrayList<>();
            meta.addAll(ds.getExampleMetaData());
        }
        if (meta != null) {
            dataset.setExampleMetaData(meta);
        }

        return dataset;*/
    }



    private static INDArray[] mergeTimeSeries(INDArray[] data, INDArray[] mask) {
        if (data.length == 1)
            return new INDArray[] {data[0], (mask == null ? null : mask[0])};

        //Complications with time series:
        //(a) They may have different lengths (if so: need input + output masking arrays)
        //(b) Even if they are all the same length, they may have masking arrays (if so: merge the masking arrays too)

        int firstLength = data[0].size(2);
        int maxLength = firstLength;

        boolean lengthsDiffer = false;
        int totalExamples = 0;
        for (INDArray arr : data) {
            int thisLength = arr.size(2);
            maxLength = Math.max(maxLength, thisLength);
            if (thisLength != firstLength)
                lengthsDiffer = true;

            totalExamples += arr.size(0);
        }

        boolean needMask = mask != null || lengthsDiffer;

        int vectorSize = data[0].size(1);

        INDArray out = Nd4j.create(new int[] {totalExamples, vectorSize, maxLength}, 'f'); //F order: better strides for time series data
        INDArray outMask = (needMask ? Nd4j.create(totalExamples, maxLength) : null);

        int rowCount = 0;

        if (!needMask) {
            //Simplest case: no masking arrays, all same length
            INDArrayIndex[] indexes = new INDArrayIndex[3];
            indexes[1] = all();
            indexes[2] = all();
            for (INDArray arr : data) {
                int nEx = arr.size(0);
                indexes[0] = interval(rowCount, rowCount + nEx);
                out.put(indexes, arr);
                rowCount += nEx;
            }
        } else {
            //Different lengths, and/or mask arrays
            INDArrayIndex[] indexes = new INDArrayIndex[3];
            indexes[1] = all();

            for (int i = 0; i < data.length; i++) {
                INDArray arr = data[i];
                int nEx = arr.size(0);
                int thisLength = arr.size(2);
                indexes[0] = interval(rowCount, rowCount + nEx);
                indexes[2] = interval(0, thisLength);
                out.put(indexes, arr);

                //Need to add a mask array...
                if (mask != null && mask[i] != null) {
                    //By merging the existing mask array

                    outMask.put(new INDArrayIndex[] {interval(rowCount, rowCount + nEx), interval(0, thisLength)},
                        mask[i]);
                } else {
                    //Because of different length data
                    outMask.get(new INDArrayIndex[] {interval(rowCount, rowCount + nEx), interval(0, thisLength)})
                        .assign(1.0);
                }

                rowCount += nEx;
            }
        }

        return new INDArray[] {out, outMask};
    }

    private static INDArray merge5dCnnData(INDArray[] data) {
        if (data.length == 1)
            return data[0];

        int[] outSize = new int[5]; //[examples,regions,depth,width,height]

        for (int i = 1; i < data[0].shape().length; i++) {
            outSize[i+1] = data[0].shape()[i];
        }

        for (int i = 0; i < data.length; i++) {
            outSize[0] += 1;
            //Attention, images hasn't same region number. Region dimesion is set to max.
            outSize[1] = Integer.max(outSize[1], data[i].shape()[0]);
        }

        INDArray out = Nd4j.create(outSize, 'c');
        int examplesSoFar = 0;
        INDArrayIndex[] indexes = new INDArrayIndex[5];

        indexes[2] = all();
        indexes[3] = all();
        indexes[4] = all();

        for (int i = 0; i < data.length; i++) {
            //Check shapes:
            int[] thisShape = data[i].shape();
            if (thisShape.length != 4)
                throw new IllegalStateException("Cannot merge CNN data: first DataSet data has shape "
                    + Arrays.toString(data[0].shape()) + ", " + i + "th example has shape "
                    + Arrays.toString(thisShape));
            /*for (int j = 1; j < 4; j++) {
                if (outSize[j] != thisShape[j])
                    throw new IllegalStateException("Cannot merge CNN data: first DataSet data has shape "
                        + Arrays.toString(data[0].shape()) + ", " + i + "th example has shape "
                        + Arrays.toString(thisShape));
            }*/

            int thisNumExamples = 1;
            //Put:
            indexes[1] = interval(0, 0 + data[i].size(0));
            indexes[0] = interval(examplesSoFar, examplesSoFar + thisNumExamples);
            out.put(indexes, data[i]);

            examplesSoFar += thisNumExamples;
        }

        return out;
    }



    private static INDArray merge4dCnnData(INDArray[] data) {
        if (data.length == 1)
            return data[0];
        int[] outSize = Arrays.copyOf(data[0].shape(), 4); //[examples,depth,width,height]

        for (int i = 1; i < data.length; i++) {
            outSize[0] += data[i].size(0);
        }

        INDArray out = Nd4j.create(outSize, 'c');
        int examplesSoFar = 0;
        INDArrayIndex[] indexes = new INDArrayIndex[4];
        indexes[1] = all();
        indexes[2] = all();
        indexes[3] = all();
        for (int i = 0; i < data.length; i++) {
            //Check shapes:
            int[] thisShape = data[i].shape();
            if (thisShape.length != 4)
                throw new IllegalStateException("Cannot merge CNN data: first DataSet data has shape "
                    + Arrays.toString(data[0].shape()) + ", " + i + "th example has shape "
                    + Arrays.toString(thisShape));
            for (int j = 1; j < 4; j++) {
                if (outSize[j] != thisShape[j])
                    throw new IllegalStateException("Cannot merge CNN data: first DataSet data has shape "
                        + Arrays.toString(data[0].shape()) + ", " + i + "th example has shape "
                        + Arrays.toString(thisShape));
            }

            int thisNumExamples = data[i].size(0);
            //Put:
            indexes[0] = interval(examplesSoFar, examplesSoFar + thisNumExamples);
            out.put(indexes, data[i]);

            examplesSoFar += thisNumExamples;
        }

        return out;
    }


    private static INDArray merge2d(INDArray[] data) {
        if (data.length == 0)
            return data[0];
        int totalRows = 0;
        for (INDArray arr : data)
            totalRows += arr.rows();
        INDArray out = Nd4j.create(totalRows, data[0].columns());

        totalRows = 0;
        for (INDArray i : data) {
            if (i.size(0) == 1)
                out.putRow(totalRows++, i);
            else {
                out.put(new INDArrayIndex[] {interval(totalRows, totalRows + i.size(0)), all()}, i);
                totalRows += i.size(0);
            }
        }
        return out;
    }

    private static INDArray mergeWithAdd3d(INDArray[] data) {
        if (data.length == 1)
            return data[0];

        int[] outSize = new int[3];

        outSize[0] = data.length;
        for (int i = 0; i < data.length; i++) {
            outSize[1] = Integer.max(outSize[1], data[i].size(0));
        }
        outSize[2] = data[0].size(1);

        INDArray out = Nd4j.create(outSize, 'c');

        int examplesSoFar = 0;
        INDArrayIndex[] indexes = new INDArrayIndex[3];

        indexes[2] = all();

        for (int i = 0; i < data.length; i++) {
            int thisNumExamples = 1;
            //Put:
            indexes[1] = interval(0, 0 + data[i].size(0));
            indexes[0] = interval(examplesSoFar, examplesSoFar + thisNumExamples);
            out.put(indexes, data[i]);

            examplesSoFar += thisNumExamples;
        }

        return out;

    }

    private static INDArray merge3d(INDArray[] data) {
        if (data.length == 0)
            return data[0];
        int totalRows = 0;
        for (INDArray arr : data)
            totalRows += arr.rows();
        INDArray out = Nd4j.create(totalRows, data[0].columns());

        totalRows = 0;
        for (INDArray i : data) {
            if (i.size(0) == 1)
                out.putRow(totalRows++, i);
            else {
                out.put(new INDArrayIndex[] {interval(totalRows, totalRows + i.size(0)), all()}, i);
                totalRows += i.size(0);
            }
        }
        return out;
    }
}
