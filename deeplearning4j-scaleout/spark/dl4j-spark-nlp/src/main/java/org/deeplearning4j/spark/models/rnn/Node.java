package org.deeplearning4j.spark.models.rnn;

public class Node extends Tree {
    public Tree left = null;
    public Tree right = null;

    public Node(Tree left, Tree right) {
        this.left = left;
        this.right = right;
    }
    @Override
    public void clearCache() {
        super.clearCache();
        if (left != null) left.clearCache();
        if (right != null) right.clearCache();
    }
}

