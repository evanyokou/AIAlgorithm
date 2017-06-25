package com.yh.ai.algorithm.bpnn;

/**
 * Created by Ypc on 2017/06/25.
 */
public class NerveNet {
    // 神経ネットワークについての節点の数の情報
    // 例えば、[2,3,4,2]とは輸入層の節点の数は２、輸出層の節点の数は２、隠す層が２層があり、
    // 第一層目の節点の数は３、第二層目の節点の数は４です
    public int[] net_info;
    // 神経ネットワークって変数です。
    public Layer[] net;
    // ネットの学習係数っていうものです。
    public Double learn_ratio;
    // 節点が自身の各権量を調整する時に値の収束を加速するために設定する。
    public Double momentum_ratio;

    /**
     * 神経ネットワークを初期化する
     * @param net_info　ネットの層数と各層の節点に関する情報。
     * @param learn_ratio　ネットの学習係数
     * @param momentum_ratio　節点が自身の各権量の調整率
     */
    public NerveNet(int[] net_info, Double learn_ratio, Double momentum_ratio){
        this.momentum_ratio = momentum_ratio;
        this.learn_ratio = learn_ratio;
        this.net_info = net_info;
        this.net = new Layer[this.net_info.length];
        for (int i = 0; i < this.net_info.length; ++i){
            // Y = w1 * x1 + w2 * x2 + ... + wn * xn + b
            //  (n)個節点の上で 一つ切片節点を添えて共(n+1)個節点

            if (i == this.net_info.length-1){
                // 輸出層
                this.net[i] = new Layer(this.net_info[i],this.net_info[i-1],0);
            }else if (i == 0){
                //輸入層
                this.net[i] = new Layer(this.net_info[i],0,this.net_info[i+1]);
            }else {
                //隠す層
                this.net[i] = new Layer(this.net_info[i],this.net_info[i-1],this.net_info[i+1]);
            }
        }
    }

    /**
     * 次第で各層節点の結果を計算する
     * @param input　　入力情報
     */
    public void compute(double[] input){
        if (input.length != this.net[0].cur_layer_node_num){
            System.err.println("輸入した値はネットの輸入層の個数とが合わないです。");
            System.err.println("必要入力："+ (this.net[0].cur_layer_node_num) +"実際入力："+ input.length);
        }else {
            //入力した情報を相応しい輸入層節点に嵌る
            for (int i=0; i<this.net[0].cur_layer_node_num; ++i){
                this.net[0].nodes[i].value = input[i];
            }
            //計算し始める
            for (int cur_layer=1; cur_layer<this.net_info.length; ++cur_layer){
                for (int cur_node=0; cur_node<this.net[cur_layer].cur_layer_node_num; ++cur_node){
                    //切片節点　b = w` * 1.0
                    double wx = 1.0 * this.net[cur_layer-1].nodes[this.net[cur_layer].pre_layer_node_num].weight[cur_node];
                    //非切片節点　wx = w0*x0 + w1*x1 + ... wn*xn
                    for (int pre_node=0; pre_node<this.net[cur_layer].pre_layer_node_num; ++pre_node){
                        wx += this.net[cur_layer-1].nodes[pre_node].value * this.net[cur_layer-1].nodes[pre_node].weight[cur_node];
                    }
                    // Y = f( wx + b )
                    this.net[cur_layer].nodes[cur_node].value = NerveNet.sigmoid(wx);
                }
            }
        }
    }

    /**
     * 誤差が逆向にいってから各節点の権量を訂正する
     * @param output
     */
    public void update(double[] output){
        int last_layer = this.net_info.length-1;
        if (output.length != this.net[last_layer].cur_layer_node_num){
            System.err.println("輸出した値はネットの輸出層の個数とが合わないです。");
            System.err.println("必要入力："+ (this.net[last_layer].cur_layer_node_num) +"実際入力："+ output.length);
        }else {
            //輸出層の各節点の値と実際の結果一斉に計算して誤差を取る
            for (int last_node=0; last_node<this.net[this.net_info.length-1].cur_layer_node_num; ++last_node){
                this.net[last_layer].nodes[last_node].error = this.net[last_layer].nodes[last_node].value *
                        (1 - this.net[last_layer].nodes[last_node].value) *
                        (output[last_node] - this.net[last_layer].nodes[last_node].value);
            }
            //隠す層と輸入層の各節点の権量を調整する
            //循環して各節点を検索する
            for (int cur_layer=last_layer-1; cur_layer>-1; cur_layer--){
                for (int cur_node=0; cur_node<this.net[cur_layer].cur_layer_node_num; ++cur_node){
                    double we = 0;
                    //普通節点
                    for (int next_node=0; next_node<this.net[cur_layer].next_layer_node_num; ++next_node){
                        we = this.net[cur_layer].nodes[cur_node].weight[next_node] * this.net[cur_layer+1].nodes[next_node].error;
                        //普通節点権量を修正する
                        this.net[cur_layer].nodes[cur_node].weight_delta[next_node] = this.momentum_ratio *
                                this.net[cur_layer].nodes[cur_node].weight_delta[next_node] +
                                this.learn_ratio * this.net[cur_layer].nodes[cur_node].value *
                                        this.net[cur_layer+1].nodes[next_node].error;
                        this.net[cur_layer].nodes[cur_node].weight[next_node] += this.net[cur_layer].nodes[cur_node].weight_delta[next_node];
                        if (cur_node == this.net[cur_layer].cur_layer_node_num-1){
                            //切片節点権量を修正する
                            this.net[cur_layer].nodes[cur_node+1].weight_delta[next_node] = this.momentum_ratio *
                                    this.net[cur_layer].nodes[cur_node+1].weight_delta[next_node] +
                                    this.learn_ratio * this.net[cur_layer+1].nodes[next_node].error;
                            this.net[cur_layer].nodes[cur_node+1].weight[next_node] += this.net[cur_layer].nodes[cur_node+1].weight_delta[next_node];
                        }
                    }
                    this.net[cur_layer].nodes[cur_node].error = we *
                            this.net[cur_layer].nodes[cur_node].value *
                            (1 - this.net[cur_layer].nodes[cur_node].value);
                }
            }
        }
    }

    public void train(double[] input,double[] output){
        compute(input);
        update(output);
    }

    public void print(double[] input){
        System.out.println("["+input[0]+","+input[1]+"]:[" + this.net[this.net_info.length-1].nodes[0].value + "," + this.net[this.net_info.length-1].nodes[1].value + "]");
    }

    public static Double sigmoid(Double param){
        return 1/(1 + Math.exp(-param));
    }

    public static double hyperbolicTangent(double param){
        return (Math.exp(param) - Math.exp(-param))/(Math.exp(param) + Math.exp(-param));
    }
}
