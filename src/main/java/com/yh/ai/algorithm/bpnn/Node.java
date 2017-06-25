package com.yh.ai.algorithm.bpnn;

import java.util.Random;

/**
 * Created by Ypc on 2017/06/25.
 */
public class Node {
    //節点の輸出値
    public double value;
    //節点の輸出値と目標値の誤差
    public double error;
    // 当層の節点と先立つ層の各節点相応しい権量
    public double[] weight;
    //節点の各権量の調整する幅の値
    public double[] weight_delta;

    /**
     * 輸出層の初期化
     */
    public Node(){
        value = 0;
        error = 0;
        weight = new double[0];
        weight_delta = new double[0];
    }

    /**
     * 普通な節点と単位節点の初期化
     * @param next_layer_node_num
     * @param intercrept   切片って節点の標識、true：切片節点 false：普通節点
     */
    public Node(int next_layer_node_num,boolean intercrept){
        error = 0;
        weight = new double[next_layer_node_num];
        weight_delta = new double[next_layer_node_num];
        Random random = new Random();
        for (int i=0; i<next_layer_node_num;++i){
            weight[i] = random.nextDouble();
            weight_delta[i] = 0;
        }
        //切片節点
        if (intercrept){
            value = 1.0;
        }else {
            //普通節点
            value = 0.0;
        }
    }
}
