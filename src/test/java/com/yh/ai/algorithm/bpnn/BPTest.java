package com.yh.ai.algorithm.bpnn;

import org.junit.Test;

/**
 * Created by Ypc on 2017/06/24.
 */
public class BPTest {

    @Test
    public void test1(){
        NerveNet nn = new NerveNet(new int[]{2,10,2},0.15,0.8);

        double[][] data = new double[][]{{1,2},{2,2},{1,1},{2,1}};
        double[][] target = new double[][]{{1,0},{0,1},{0,1},{1,0}};

        for(int n=0;n<1000;n++)
            for(int i=0;i<data.length;i++) {
                nn.train(data[i],target[i]);
                nn.print(data[i]);
            }

        nn.compute(new double[]{3,1});
        nn.print(new double[]{3,1});
        nn.compute(new double[]{3,3});
        nn.print(new double[]{3,3});
    }
}
