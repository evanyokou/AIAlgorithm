package com.yh.ai.algorithm.ga;

import java.util.Random;

/**
 * Created by Ypc on 2017/06/25.
 */
public class Gene {
    //基因の長さ
    public int length;
    //基因のチェーン
    public double[] chain;

    public Gene(int length,double max,double min){
        this.length = length;
        this.chain = new double[this.length];
        Random r = new Random();
        for (int i=0; i<this.length; ++i){
            this.chain[i] = min + (max - min)*r.nextDouble();
        }
    }
}
