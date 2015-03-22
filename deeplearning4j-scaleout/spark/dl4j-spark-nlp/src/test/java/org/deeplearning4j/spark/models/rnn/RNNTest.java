package org.deeplearning4j.spark.models.rnn;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * Created by tomasz on 3/22/15.
 */
public class RNNTest {

    protected Tree[] sampleTree() {
        double[] vl1 = {-1.0, 1.0};
        double[] vl2 = {-2.0, -2.0};
        double[] vl3 = {3.0, 5.0};

        Tree l1 = new Leaf(Nd4j.create(vl1).transpose());
        Tree l2 = new Leaf(Nd4j.create(vl2).transpose());
        Tree l3 = new Leaf(Nd4j.create(vl3).transpose());

        Tree n1 = new Node(l1, l2);
        Tree n2 = new Node(n1, l3);

        Tree[] nodes = {l1, l2, l3, n1, n2};

        return nodes;
    }

    protected INDArray sampleCombinator() {
        double[][] vc = {{1.0, -2.0, -3.0, 1.0, 1.0}, {2.0, -3.0, 1.0, 2.0, -2.0}};
        INDArray combinator = Nd4j.create(vc);
        return combinator;
    }

    protected INDArray sampleJudge() {
        double[][] vj = {{11.0, 13.0}, {17.0, -19.0}, {-3.0, -2.0}};
        INDArray judge = Nd4j.create(vj);
        return judge;
    }

    @Test
    public void testFoldTree() throws Exception {
        Tree[] nodes = sampleTree();
        INDArray combinator = sampleCombinator();
        RNN.Settings settings = new RNN.Settings(2);
        settings.layerActivation = "abs";
        RNN rnn = new RNN(settings, combinator, null);

        rnn.foldTree(nodes[4]);

        double expected[][] = {
            {1.0, 1.0},
            {2.0, 2.0},
            {3.0, 5.0},
            {4.0, 3.0},
            {5.0, 10.0},
        };

        for(int i = 0; i < nodes.length; i++)
            for(int j = 0; j < expected[0].length; j++)
                assertEquals(expected[i][j], nodes[i].cache.getDouble(j), 0.0);
    }


}
