package org.deeplearning4j.spark.models.rnn;

import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
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
    public void fit(List<Pair<Tree<INDArray>, INDArray>> trees) {

    }

    /**
     * Forward propagates the network and returns the result.
     * @param tree The tree.
     * @return The resulting vector of probabilities for each class.
     */
    public INDArray predict(Tree<INDArray> tree) {
        INDArray vec = foldTree(tree);
        INDArray outVec = judge.mulColumnVector(vec);
        assert isValidOutVec(outVec);

        INDArray result = transform(settings.outputActivation, outVec);
        assert isValidOutVec(result);
        return result;
    }

    /**
     * Folds a tree using the combinator.
     * @param tree The tree to fold.
     * @return An input vector for the entire tree.
     */
    public INDArray foldTree(Tree<INDArray> tree) {
        if (tree instanceof Leaf<?>) {
            INDArray value = ((Leaf<INDArray>) tree).value;
            assert isValidInVec(value);
            return value;
        } else {
            Node<INDArray> node = (Node<INDArray>) tree;
            INDArray left = foldTree(node.left);
            INDArray right = foldTree(node.right);
            assert isValidInVec(left) && isValidInVec(right);

            INDArray combined = Nd4j.appendBias(left, right);
            INDArray vec = combinator.mulColumnVector(combined); // local der: transponuj wektor "z nieba" i rozszerz wertykalnie (np. mnożąc przez wektor jedynek)
            assert isValidInVec(vec);

            INDArray result = transform(settings.layerActivation, vec); //
            assert isValidInVec(result);
            return result;
        }
    }

    /**
     * Computes the error.
     * @param tree A sample tree.
     * @param expected Expected result on this sample.
     * @return The error.
     */
    public double error(Tree<INDArray> tree, INDArray expected) {
        assert isValidOutVec(expected);
        INDArray actual = predict(tree);
        INDArray logActual = transform("log", actual);

        double positive = Nd4j.getBlasWrapper().dot(
                Nd4j.ones(settings.outSize, 1).subi(expected), logActual);
        double negative = Nd4j.getBlasWrapper().dot(expected, logActual);

        double combinatorReg = accumulate("norm2", combinator);
        double judgeReg = accumulate("norm2", judge);
        double regularization = combinatorReg * combinatorReg + judgeReg * judgeReg; // der: 2 * M

        return positive - negative + settings.regularizationCoeff * regularization;
    }

    protected INDArray transform(String how, INDArray what) {
        TransformOp transformation = Nd4j.getOpFactory().createTransform(how, what);
        return Nd4j.getExecutioner().execAndReturn(transformation);
    }

    protected double accumulate(String how, INDArray what) {
        Accumulation accumulation = Nd4j.getOpFactory().createAccum(how, what);
        return Nd4j.getExecutioner().execAndReturn(accumulation).currentResult().doubleValue();
    }

    public boolean isValidInVec(INDArray vec) {
        return vec.isColumnVector() && vec.rows() == settings.inSize;
    }

    public boolean isValidOutVec(INDArray vec) {
        return vec.isColumnVector() && vec.rows() == settings.outSize;
    }
}
