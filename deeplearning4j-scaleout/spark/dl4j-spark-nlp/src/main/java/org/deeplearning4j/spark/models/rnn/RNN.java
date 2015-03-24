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
    public static class Settings implements Serializable {
        public Settings(int inSize) {
            this.inSize = inSize;
            this.noise = 0.01;
            this.rng = new DefaultRandom();
            this.layerActivation = "tanh";
            this.outputActivation = "softmax";
            this.regularizationCoeff = 0.001;
            this.learningRate = 0.001;
        }

        public int inSize;
        public int outSize;
        public double noise;
        public Random rng;
        public String layerActivation;
        public String outputActivation;
        public double regularizationCoeff;
        public double learningRate;
    }

    public INDArray combinator;
    public INDArray judge;
    public Settings settings;

    /**
     * Creates a clean RNN.
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
     * Creates RNN with specified combinator and judge.
     * @param settings
     * @param combinator
     * @param judge
     */
    public RNN(Settings settings, INDArray combinator, INDArray judge) {
        this.settings = settings;
        this.combinator = combinator;
        this.judge = judge;
    }

    /**
     * Trains the network on a set of trees.
     * @param labeledTrees The trees with labels.
     */
    public void fit(List<Pair<Tree, INDArray>> labeledTrees) {
        for(Pair<Tree, INDArray> labeledTree : labeledTrees) {
            Tree tree = labeledTree.getFirst();
            INDArray label = labeledTree.getSecond();
            tree.clearCache();
            Double err = error(tree, label);
            INDArray combinatorGradient = errorGradWithRespectToCombinator(tree, label);
            INDArray judgeGradient = errorGradWithRespectToJudge(tree, label);
            combinator.subi(combinatorGradient.mul(settings.learningRate));
            judge.subi(judgeGradient.mul(settings.learningRate));
        }
    }

    /**
     * Folds a tree using the combinator. Stores result in tree cache.
     * @param tree The tree to fold.
     * @return An input vector for the entire tree.
     */
    public INDArray foldTree(Tree tree) {
        if (tree.cache == null) {
            if (tree instanceof Leaf) {
                INDArray value = ((Leaf) tree).value;
                assert isValidInVec(value);

                INDArray transformed = transform(settings.layerActivation, value);
                assert isValidInVec(transformed);
                tree.cache = transformed;
            } else {
                Node node = (Node) tree;
                INDArray left = foldTree(node.left);
                INDArray right = foldTree(node.right);
                assert isValidInVec(left) && isValidInVec(right);

                INDArray combined = Nd4j.appendBias(left, right);
                INDArray vec = combinator.mmul(combined);
                assert isValidInVec(vec);

                INDArray transformed = transform(settings.layerActivation, vec);
                assert isValidInVec(transformed);
                tree.cache = transformed;
            }
        }
        return tree.cache;
    }

    /**
     * Calculates gradient of foldTree with respect to combinator.
     * @param tree
     * @return (i, j, k) -> gradient of foldTree_k function with respect to combinator_ij.
     */
    public INDArray[][] foldTreeGradWithRespectToCombinator(Tree tree) {
        if (tree instanceof Leaf) {
            INDArray[][] gradient = emptyGradient();
            for(int i = 0; i < combinator.rows(); i++)
                for(int j = 0; j < combinator.columns(); i++) {
                    INDArray vec = Nd4j.zeros(combinator.rows());
                    gradient[i][j] = vec;
                }
            return gradient;
        } else {
            Node node = (Node) tree;
            INDArray[][] leftGradient = foldTreeGradWithRespectToCombinator(node.left);
            INDArray[][] rightGradient = foldTreeGradWithRespectToCombinator(node.right);
            INDArray leftValue = foldTree(node.left);
            INDArray rightValue = foldTree(node.right);
            INDArray nodeValue = foldTree(node);
            assert isValidInVec(leftValue) && isValidInVec(rightValue) && isValidInVec(nodeValue);

            // layerActivation'(combinator(treeVal))
            INDArray nodeValueTWD = transformWithDerivative(settings.layerActivation, nodeValue);
            // [leftValue rightValue 1]
            INDArray v = Nd4j.appendBias(leftValue, rightValue);

            INDArray[][] gradient = emptyGradient();
            for(int i = 0; i < combinator.rows(); i++)
                for(int j = 0; j < combinator.columns(); j++) {
                    // [leftValue' rightValue' 0]
                    INDArray vPrime = Nd4j.vstack(leftGradient[i][j], rightGradient[i][j], Nd4j.zeros(1));
                    INDArray nodeValuePrime = combinator.mulColumnVector(vPrime);
                    nodeValuePrime = nodeValuePrime.putScalar(i, nodeValuePrime.getDouble(i) + v.getDouble(j));
                    nodeValuePrime = nodeValuePrime.mul(nodeValueTWD);
                    assert isValidOutVec(nodeValuePrime);

                    gradient[i][j] = nodeValuePrime;
                }
            return gradient;
        }
    }

    /**
     * Forward propagates the network and returns the result.
     * @param tree
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
     * @param tree
     * @return (i, j, k) -> gradient of predict_k function with respect to combinator_ij
     */
    public INDArray[][] predictGradWithRespectToCombinator(Tree tree) {
        INDArray[][] treeGradient = foldTreeGradWithRespectToCombinator(tree);

        INDArray treeValue = foldTree(tree);
        assert isValidInVec(treeValue);

        // judge(treeValue)
        INDArray v = judge.mulColumnVector(treeValue);
        assert isValidOutVec(v);

        // outputActivation'(judge(treeValue))
        INDArray w = transformWithDerivative(settings.outputActivation, v);
        assert isValidOutVec(w);

        INDArray[][] gradient = emptyGradient();
        for(int i = 0; i < combinator.rows(); i++)
            for(int j = 0; j < combinator.columns(); j++) {
                // treeValue'
                INDArray treeValuePrime = treeGradient[i][j];

                INDArray vPrime = judge.mulColumnVector(treeValuePrime);
                vPrime = vPrime.mmul(w);
                assert isValidOutVec(vPrime);

                gradient[i][j] = vPrime;
            }

        return gradient;
    }

    /**
     * Calculates gradient of predict function with respect to judge elements.
     * @param tree
     * @return Gradient matrix.
     */
    public INDArray predictGradWithRespectToJudge(Tree tree) {
        INDArray treeValue = foldTree(tree);
        assert isValidInVec(treeValue);

        // judge(treeValue)
        INDArray v = judge.mul(treeValue);

        // (judge(treeValue))'
        INDArray vPrime = treeValue.transpose().mul(Nd4j.ones(judge.rows(), 1));

        // outputActivation'(judge(treeValue))
        INDArray vTWD = transformWithDerivative(settings.outputActivation, v);

        INDArray gradient = v.mmul(vTWD);

        return gradient;
    }

    /**
     * Computes the error.
     * @param tree A sample tree.
     * @param expected Expected result on this sample.
     * @return The error.
     */
    public double error(Tree tree, INDArray expected) {
        INDArray actual = predict(tree);
        assert isValidOutVec(expected) && isValidOutVec(actual);

        INDArray logActual = transform("log", actual);

        double positive = Nd4j.getBlasWrapper().dot(
                Nd4j.ones(settings.outSize, 1).subi(expected), logActual);
        double negative = Nd4j.getBlasWrapper().dot(expected, logActual);

        double combinatorReg = accumulate("norm2", combinator);
        double judgeReg = accumulate("norm2", judge);
        double regularization = combinatorReg * combinatorReg + judgeReg * judgeReg;

        return positive - negative + settings.regularizationCoeff * regularization;
    }

    /**
     * Calculates gradient of error function with respect to combinator elements.
     * @param tree
     * @param expected Expected result.
     * @return Gradient matrix.
     */
    public INDArray errorGradWithRespectToCombinator(Tree tree, INDArray expected) {
        INDArray actual = predict(tree);
        assert isValidOutVec(expected) && isValidOutVec(actual);

        INDArray[][] predictGradient = predictGradWithRespectToCombinator(tree);

        INDArray[][] gradient = emptyGradient();

        // log
        INDArray actualTWD = transformWithDerivative("log", actual);
        for(int i = 0; i < combinator.rows(); i++)
            for(int j = 0; j < combinator.columns(); j++) {
                INDArray logActualPrime = predictGradient[i][j];
                logActualPrime = logActualPrime.mmul(actualTWD);
                gradient[i][j] = logActualPrime;
            }

        // dot product
        INDArray expectedT = expected.transpose();
        INDArray onesMinusExpectedT = Nd4j.ones(1, settings.outSize).subi(expectedT);
        for(int i = 0; i < combinator.rows(); i++)
            for(int j = 0; j < combinator.columns(); j++) {
                INDArray gradientIJ = gradient[i][j];
                INDArray positive = onesMinusExpectedT.mulColumnVector(gradientIJ);
                INDArray negative = expectedT.mulColumnVector(gradientIJ);
                INDArray substraction = positive.sub(negative);
                gradient[i][j] = substraction;
            }

        // bias
        INDArray flatGradient = combinator.mul(2.0);

        // flatten the result
        for(int i = 0; i < combinator.rows(); i++)
            for(int j = 0; j < combinator.columns(); j++) {
                Double tmp1 = flatGradient.getDouble(i, j);
                Double tmp2 = gradient[i][j].getDouble(0);
                flatGradient.put(i, j, tmp1 + tmp2);
            }

        return flatGradient;
    }

    /**
     * Calculates gradient of predict function with respect to judge elements.
     * @param tree
     * @param expected Expected result.
     * @return Gradient matrix.
     */
    public INDArray errorGradWithRespectToJudge(Tree tree, INDArray expected) {
        INDArray actual = predict(tree);
        assert isValidOutVec(expected) && isValidOutVec(actual);

        INDArray predictGradient = predictGradWithRespectToJudge(tree);
        INDArray actualTWD = transformWithDerivative("log", actual);
        INDArray gradient = predictGradient.mmul(actualTWD);
        INDArray positive = gradient.mmul(expected);
        INDArray negative = gradient.mmul(Nd4j.ones(settings.outSize, 1).subi(expected));
        gradient = positive.sub(negative);

        return gradient;
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

    protected INDArray[][] emptyGradient() {
        INDArray[][] map = new INDArray[combinator.rows()][combinator.columns()];
        return map;
    }

    public boolean isValidInVec(INDArray vec) {
        return vec.isColumnVector() && vec.rows() == settings.inSize;
    }

    public boolean isValidOutVec(INDArray vec) {
        return vec.isColumnVector() && vec.rows() == settings.outSize;
    }
}
