package org.deeplearning4j.examples.deeplogo;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.eval.ConfusionMatrix;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by andlatel on 24/05/2017.
 */
public class MyEvaluation extends Evaluation {
    protected final int topN;
    protected int topNCorrectCount = 0;
    protected int topNTotalCount = 0; //Could use topNCountCorrect / (double)getNumRowCounter() - except for eval(int,int), hence separate counters
    protected Counter<Integer> truePositives = new Counter<>();
    protected Counter<Integer> falsePositives = new Counter<>();
    protected Counter<Integer> trueNegatives = new Counter<>();
    protected Counter<Integer> falseNegatives = new Counter<>();
    protected ConfusionMatrix<Integer> confusion;
    protected int numRowCounter = 0;
    protected int rightLabel = 0;
    @Getter
    @Setter
    protected List<String> labelsList = new ArrayList<>();
    //What to output from the precision/recall function when we encounter an edge case
    protected static final double DEFAULT_EDGE_VALUE = 0.0;

    protected Map<Pair<Integer, Integer>, List<Object>> confusionMatrixMetaData; //Pair: (Actual,Predicted)

    public MyEvaluation() {
        topN = 1;
    }

    public MyEvaluation(int numClasses) {
        this(createLabels(numClasses), 1);
    }

    public MyEvaluation(List<String> labels, int topN) {
        this.labelsList = labels;
        if (labels != null) {
            createConfusion(labels.size());
        }
        this.topN = topN;
    }

    private static List<String> createLabels(int numClasses) {
        if (numClasses == 1)
            numClasses = 2; //Binary (single output variable) case...
        List<String> list = new ArrayList<>(numClasses);
        for (int i = 0; i < numClasses; i++) {
            list.add(String.valueOf(i));
        }
        return list;
    }

    @Override
    public void eval(INDArray realOutcomes, INDArray guesses, List<? extends Serializable> recordMetaData) {
        // Add the number of rows to numRowCounter
        numRowCounter += realOutcomes.shape()[0];

        // If confusion is null, then Evaluation was instantiated without providing the classes -> infer # classes from
        if (confusion == null) {
            int nClasses = realOutcomes.columns();
            if (nClasses == 1)
                nClasses = 2; //Binary (single output variable) case
            labelsList = new ArrayList<>(nClasses);
            for (int i = 0; i < nClasses; i++)
                labelsList.add(String.valueOf(i));
            createConfusion(nClasses);
        }

        // Length of real labels must be same as length of predicted labels
        if (realOutcomes.length() != guesses.length())
            throw new IllegalArgumentException("Unable to evaluate. Outcome matrices not same length");

        // For each row get the most probable label (column) from prediction and assign as guessMax
        // For each row get the column of the true label and assign as currMax

        int nCols = realOutcomes.columns();
        int nRows = realOutcomes.rows();

        if (nCols == 1) {}
        else {
            INDArray guessIndex = Nd4j.argMax(guesses, 1);
            INDArray realOutcomeIndex = Nd4j.argMax(realOutcomes, 1);

            int nExamples = guessIndex.length();
            for (int i = 0; i < nExamples; i++) {
                int col = (int)realOutcomeIndex.getInt(i);
                if( guesses.getInt(i, col) == 1) rightLabel++;
            }
        }
    }

    private void createConfusion(int nClasses) {
        List<Integer> classes = new ArrayList<>();
        for (int i = 0; i < nClasses; i++) {
            classes.add(i);
        }

        confusion = new ConfusionMatrix<>(classes);
    }

    private String resolveLabelForClass(Integer clazz) {
        if (labelsList != null && labelsList.size() > clazz)
            return labelsList.get(clazz);
        return clazz.toString();
    }

    public String stats(boolean suppressWarnings) {
        StringBuilder builder = new StringBuilder().append("\n");
        StringBuilder warnings = new StringBuilder();

        builder.append("\n");
        builder.append(warnings);

        DecimalFormat df = new DecimalFormat("#.####");

        builder.append("\n==========================Scores========================================");
        builder.append("\n Accuracy:        ").append(format(df, rightLabel /(double) numRowCounter));
        //builder.append("\n Precision:       ").append(format(df, prec));
        //builder.append("\n Recall:          ").append(format(df, rec));
        //builder.append("\n F1 Score:        ").append(format(df, f1));
        builder.append("\n========================================================================");
        return builder.toString();
    }

    private static String format(DecimalFormat f, double num) {
        if (Double.isNaN(num) || Double.isInfinite(num))
            return String.valueOf(num);
        return f.format(num);
    }

}
