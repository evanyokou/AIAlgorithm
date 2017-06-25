package com.yh.ai.algorithm.ga;

/**
 * Created by Ypc on 2017/06/25.
 */
public class Population {
    //種群の規模
    public int size;
    //基因の長さ
    public int length;
    //個体の配列
    public Gene[] individual;
    //全体に一番優れる個体
    public Gene best_individual;
    //学習率
    public double learn_ratio = 0.02;
    //基因交差率
    public double variant_ratio = 0.8;
    //突然変異率
    public double mutant_ratio = 0.2;
    //基因の最大値
    public double max;
    //基因の最小値
    public double min;

    /**
     * 初期化
     * @param size
     * @param length
     * @param max
     * @param min
     */
    public void Population(int size,int length,double max,double min){
        this.size = size;
        this.length = length;
        this.max = max;
        this.min = min;
        this.individual = new Gene[this.size];
    }

    /**
     * 初期化
     * @param size
     * @param length
     * @param max
     * @param min
     * @param learn_ratio
     * @param variant_ratio
     * @param mutant_ratio
     */
    public void Population(int size,int length,double max,double min,double learn_ratio,double variant_ratio,double mutant_ratio){
        this.size = size;
        this.length = length;
        this.max = max;
        this.min = min;
        this.learn_ratio = learn_ratio;
        this.variant_ratio = variant_ratio;
        this.mutant_ratio = mutant_ratio;
        this.individual = new Gene[this.size];
    }
}
