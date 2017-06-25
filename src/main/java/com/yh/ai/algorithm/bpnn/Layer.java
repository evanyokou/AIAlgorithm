package com.yh.ai.algorithm.bpnn;

/**
 * Created by Ypc on 2017/06/25.
 */
public class Layer {

    //当層の節点の個数
    public int cur_layer_node_num;
    //先立つ層の節点の個数
    public int pre_layer_node_num;
    //明くる層の節点の個数
    public int next_layer_node_num;
    //当層の変量
    public Node[] nodes;

    /**
     * 各層の初期化
     * Y = w0 * x0 + w1 * x1 + ... + wn * xn + b
     * b = w` * 1.0  切片値
     * @param cur_layer_node_num　　当層の節点の個数
     * @param pre_layer_node_num　　先立つ層の節点の個数
     * @param next_layer_node_num　　次の層の節点の個数
     */
    public Layer(int cur_layer_node_num,int pre_layer_node_num,int next_layer_node_num){
        this.cur_layer_node_num = cur_layer_node_num;
        this.pre_layer_node_num = pre_layer_node_num;
        this.next_layer_node_num = next_layer_node_num;

        //輸出層の節点の初期化、それら節点は次ぐ層がないから、next_layer_node_num　はゼロです。
        if (next_layer_node_num == 0){
            this.nodes = new Node[cur_layer_node_num];
            for (int i=0; i<cur_layer_node_num;++i){
                //普通節点の初期化
                this.nodes[i] = new Node();
            }
        }else {
            //非輸出層の節点の初期化
            this.nodes = new Node[cur_layer_node_num+1];
            for (int i=0; i<cur_layer_node_num;++i){
                //普通節点の初期化
                this.nodes[i] = new Node(next_layer_node_num,false);
            }
            //切片節点の初期化
            this.nodes[cur_layer_node_num] = new Node(next_layer_node_num,true);
        }
    }

}
