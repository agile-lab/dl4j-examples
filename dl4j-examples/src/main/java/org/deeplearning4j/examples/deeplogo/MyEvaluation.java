package org.deeplearning4j.examples.deeplogo;

import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.List;

/**
 * Created by andlatel on 24/05/2017.
 */
//public class MyEvaluation extends Evaluation {
//
//    @Override
//    public void eval(INDArray realOutcomes, INDArray guesses, List<? extends Serializable> recordMetaData) {
//        INDArray realOutcomeIndex = Nd4j.argMax(realOutcomes, 1);
//
//        //for (int i = 0; i < nExamples; i++) {
//
//            INDArray predicted = guesses.mul(realOutcomeIndex);
//
//            confusion.add(actual, predicted);
//
//            if (recordMetaData != null && recordMetaData.size() > i) {
//                Object m = recordMetaData.get(i);
//                addToMetaConfusionMatrix(actual, predicted, m);
//            }
//        //}
//
//        for (int col = 0; col < nCols; col++) {
//            INDArray colBinaryGuesses = guessIndex.eps(col);
//            INDArray colRealOutcomes = realOutcomes.getColumn(col);
//
//            int colTp = colBinaryGuesses.mul(colRealOutcomes).sumNumber().intValue();
//            int colFp = colBinaryGuesses.mul(colRealOutcomes.mul(-1.0).addi(1.0)).sumNumber().intValue();
//            int colFn = colBinaryGuesses.mul(-1.0).addi(1.0).muli(colRealOutcomes).sumNumber().intValue();
//            int colTn = nRows - colTp - colFp - colFn;
//
//            truePositives.incrementCount(col, colTp);
//            falsePositives.incrementCount(col, colFp);
//            falseNegatives.incrementCount(col, colFn);
//            trueNegatives.incrementCount(col, colTn);
//        }
//
//    }
//}
