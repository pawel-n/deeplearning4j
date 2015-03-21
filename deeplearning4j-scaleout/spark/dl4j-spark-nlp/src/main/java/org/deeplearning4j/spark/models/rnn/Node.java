package org.deeplearning4j.spark.models.rnn;

public class Node extends Tree {
    public Tree left = null;
    public Tree right = null;

    @Override
    public void clearCache() {
        super.clearCache();
        if (left != null) left.clearCache();
        if (right != null) right.clearCache();
    }
}

