package org.deeplearning4j.spark.models.rnn;

import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by tomasz on 3/19/15.
 */
public class RNN implements Serializable {
    public class Settings implements Serializable {
        public Settings(int inSize) {
            this.inSize = inSize;
            this.noise = 0.01;
            this.rng = new DefaultRandom();
            this.layerActivation = "tanh";
            this.outputActivation = "softmax";
            this.regularizationCoeff = 0.001;
        }

        public int inSize;
        public int outSize;
        public double noise;
        public Random rng;
        public String layerActivation;
        public String outputActivation;
        public double regularizationCoeff;
    }

    public INDArray combinator;
    public INDArray judge;
    public Settings settings;

    /**
     * Creates a tabular rasa RNN.
     */
    public RNN(Settings settings) {
        this.settings = settings;
        this.combinator = Nd4j.rand(
                settings.inSize, // rows
                2 * settings.inSize + 1, // columns
                -settings.noise,
                settings.noise,
                settings.rng
        );
        this.judge = Nd4j.rand(
                settings.outSize,
                settings.inSize,
                -settings.noise,
                settings.noise,
                settings.rng
        );
    }

    /**
     * Trains the network on a set of trees.
     * @param trees The trees with labels.
     */
    public void fit(List<Pair<Tree, INDArray>> labeledTrees) {
        for(Pair<Tree, INDArray> labeledTree : labeledTrees) {

        }
    }

    /**
     * Folds a tree using the combinator. Stores result in tree cache.
     * @param tree The tree to fold.
     * @return An input vector for the entire tree.
     */
    public INDArray foldTree(Tree tree) {
        if (tree instanceof Leaf) {
            Leaf leaf = (Leaf) tree;
            INDArray value = leaf.value;
            assert isValidInVec(value);

            INDArray transformed = transform(settings.layerActivation, value);
            assert isValidInVec(transformed);
            leaf.cache = transformed;
            return transformed;
        } else {
            Node node = (Node) tree;
            INDArray left = foldTree(node.left);
            INDArray right = foldTree(node.right);
            assert isValidInVec(left) && isValidInVec(right);

            INDArray combined = Nd4j.appendBias(left, right);
            assert isValidVecForCombinator(combined);

            INDArray vec = combinator.mulColumnVector(combined);
            assert isValidInVec(vec);

            INDArray transformed = transform(settings.layerActivation, vec);
            assert isValidInVec(transformed);
            node.cache = transformed;
            return transformed;
        }
    }

    /**
     * Calculates gradient of foldTree with respect to combinator.
     * @param tree Tree with cached values.
     * @return (i, j, k) -> gradient of foldTree_k function with respect to combinator_ij.
     */
    public ArrayList<ArrayList<INDArray>> foldTreeGradWithRespectToCombinator(Tree tree) {
        if (tree instanceof Leaf) {
            ArrayList<ArrayList<INDArray>> gradMap = emptyCombinatorGradMap();
            for(int i = 0; i < combinator.rows(); i++)
                for(int j = 0; j < combinator.columns(); i++) {
                    INDArray zeroGradVec = Nd4j.zeros(combinator.rows());
                    assert isValidInVec(zeroGradVec);
                    gradMap.get(i).set(j, zeroGradVec);
                }
            return gradMap;
        } else {
            Node node = (Node) tree;
            ArrayList<ArrayList<INDArray>> leftGradMap =
                    foldTreeGradWithRespectToCombinator(node.left);
            ArrayList<ArrayList<INDArray>> rightGradMap =
                    foldTreeGradWithRespectToCombinator(node.right);
            INDArray leftVal = node.left.cache;
            INDArray rightVal = node.right.cache;
            INDArray nodeVal = node.cache;
            assert isValidInVec(leftVal) && isValidInVec(rightVal) && isValidInVec(nodeVal);

            // layerActivation'(combinator(treeVal))
            INDArray v1 = transformWithDerivative(settings.layerActivation, nodeVal);

            INDArray combinedVal = Nd4j.appendBias(leftVal, rightVal);
            assert isValidVecForCombinator(combinedVal);


            ArrayList<ArrayList<INDArray>> gradMap = emptyCombinatorGradMap();
            for(int i = 0; i < combinator.rows(); i++)
                for(int j = 0; j < combinator.columns(); j++) {
                    INDArray leftGrad = leftGradMap.get(i).get(j);
                    INDArray rightGrad = rightGradMap.get(i).get(j);
                    assert isValidInVec(leftGrad) && isValidInVec(rightGrad);

                    INDArray combinedGrad = Nd4j.vstack(leftGrad, rightGrad, Nd4j.zeros(1));
                    assert isValidVecForCombinator(combinedGrad);

                    INDArray v2 = combinator.mulColumnVector(combinedGrad);
                    INDArray v3 = v1.putScalar(i, v2.getDouble(i) + combinedVal.getDouble(j));
                    INDArray derivatives = v3.mmul(v1);
                    assert isValidInVec(derivatives);
                    gradMap.get(i).set(j, derivatives);
                }
            return gradMap;
        }
    }

    /**
     * Forward propagates the network and returns the result.
     * @param tree The tree.
     * @return The resulting vector of probabilities for each class.
     */
    public INDArray predict(Tree tree) {
        INDArray vec = foldTree(tree);
        INDArray outVec = judge.mulColumnVector(vec);
        assert isValidOutVec(outVec);

        INDArray result = transform(settings.outputActivation, outVec);
        assert isValidOutVec(result);
        return result;
    }

    /**
     * Calculates gradient of predict function with respect to combinator elements.
     * @param tree Cached tree.
     * @return (i, j, k) -> gradient of predict_k function with respect to combinator_ij
     */
    public ArrayList<ArrayList<INDArray>> predictGradWithRespectToCombinator(Tree tree) {
        ArrayList<ArrayList<INDArray>> treeGradMap =
                foldTreeGradWithRespectToCombinator(tree);
        INDArray treeVal = tree.cache;
        assert isValidInVec(treeVal);

        // outputActivation'(judge(treeVal))
        INDArray v1 = judge.mulColumnVector(treeVal);
        INDArray v2 = transformWithDerivative(settings.outputActivation, v1);
        assert isValidOutVec(v2);

        ArrayList<ArrayList<INDArray>> judgeGradMap = emptyCombinatorGradMap();
        for(int i = 0; i < combinator.rows(); i++)
            for(int j = 0; j < combinator.columns(); j++) {
                INDArray v3 = treeGradMap.get(i).get(j);

                INDArray v4 = judge.mulColumnVector(v3);
                INDArray derivatives = v4.mmul(v2);
                assert isValidOutVec(derivatives);
                judgeGradMap.get(i).set(j, derivatives);
            }

        return judgeGradMap;
    }

    /**
     * Computes the error.
     * @param tree A sample tree.
     * @param expected Expected result on this sample.
     * @return The error.
     */
    public double error(Tree tree, INDArray expected) {
        assert isValidOutVec(expected);
        INDArray actual = predict(tree);
        INDArray logActual = transform("log", actual);

        double positive = Nd4j.getBlasWrapper().dot(
                Nd4j.ones(settings.outSize, 1).subi(expected), logActual);
        double negative = Nd4j.getBlasWrapper().dot(expected, logActual);

        double combinatorReg = accumulate("norm2", combinator);
        double judgeReg = accumulate("norm2", judge);
        double regularization = combinatorReg * combinatorReg + judgeReg * judgeReg;

        return positive - negative + settings.regularizationCoeff * regularization;
    }

    public ArrayList<ArrayList<INDArray>> errorGradWithRespectToCombinator(Tree tree, INDArray expected) {
        assert isValidOutVec(expected);
        INDArray actual = predict(tree);
        INDArray logActual = transform("log", actual);
        ArrayList<ArrayList<INDArray>> judgeGradMap = predictGradWithRespectToCombinator(tree);

        ArrayList<ArrayList<INDArray>> gmap = emptyCombinatorGradMap();
        for(int i = 0; i < combinator.rows(); i++)
            for(int j = 0; j < combinator.columns(); i++) {
                INDArray v1 = judgeGradMap.get(i).get(j);
                INDArray v2 = v1.mmul(logActual);
                gmap.get(i).set(j, v2);
            }
// curr
    }

    protected INDArray transform(String how, INDArray what) {
        TransformOp transformation = Nd4j.getOpFactory().createTransform(how, what);
        return Nd4j.getExecutioner().execAndReturn(transformation);
    }

    protected INDArray transformWithDerivative(String function, INDArray what) {
        TransformOp transformation = Nd4j.getOpFactory().createTransform(function, what);
        return Nd4j.getExecutioner().execAndReturn(transformation.derivative());
    }

    protected double accumulate(String how, INDArray what) {
        Accumulation accumulation = Nd4j.getOpFactory().createAccum(how, what);
        return Nd4j.getExecutioner().execAndReturn(accumulation).currentResult().doubleValue();
    }

    protected ArrayList<ArrayList<INDArray>> emptyGradMap(int rows, int columns) {
        ArrayList<ArrayList<INDArray>> map = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            ArrayList<INDArray> row = new ArrayList<>();
            for (int j = 0; j < columns; j++)
                row.add(null);
            map.add(row);
        }
        return map;
    }

    protected ArrayList<ArrayList<INDArray>> emptyCombinatorGradMap() {
        return emptyGradMap(combinator.rows(), combinator.columns());
    }

    protected ArrayList<ArrayList<INDArray>> emptyJudgeGradMap() {
        return emptyGradMap(judge.rows(), judge.columns());
    }

    public boolean isValidInVec(INDArray vec) {
        return vec.isColumnVector() && vec.rows() == settings.inSize;
    }

    public boolean isValidOutVec(INDArray vec) {
        return vec.isColumnVector() && vec.rows() == settings.outSize;
    }

    public boolean isValidVecForCombinator(INDArray vec) {
        return vec.isColumnVector() && vec.rows() == combinator.rows();
    }
}
