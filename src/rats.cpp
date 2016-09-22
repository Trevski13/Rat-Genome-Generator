#include <iostream>
#include <string>
#include <vector>
#include <bitset>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <ctime>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

/* Number of random of rats to initialize */
#define INITIAL_POPULATION      30000 // min is 100 - # of seeds

/* Number of generations to run */
#define NUM_GENERATIONS         1000

/* How long the chromosome is */
#define CHROMOSOME_LENGTH       290

#define MUTATION_RATE_PERCENT   99
#define MUTATION_RATE_OUTOF     100
#define PERCENT_OF_GENE_TO_MUTATE   .344827586
#define GENE_MUTATE_OUTOF       100
#define CROSSOVER_RATE_PERCENT  35
#define NUM_COUPLES_TO_PICK     25
#define NUM_CHILDREN            1
#define NUM_RANDOM_RATS         5
#define RAT_SELECTION_METHOD    "trevor" //keith or trevor or generic
#define RAT_COMBINATION_METHOD  "keith" //keith or trevor (doesn't quite work at the moment)
#define USE_SEEDS				true
#define PRUNE_PERCENT			33 //percent of rats not safe via PRUNE_CUTOFF to prune
#define PRUNE_CUTOFF			25 //top percentage of rats that are "safe" from pruning
//int startPOS[5][2] = { {12, 12}, {5, 5}, {17, 17}, {3, 6}, {2, 21} };
//int startPOS[4][2] = { {12, 12}, {5, 5}, {17, 17}, {3, 6} };
//int startPOS[3][2] = { {12, 12}, {5, 5}, {17, 17} };
int startPOS[1][2] = { {12, 12} };
#define MULTI_MAZE_WEIGHT		"min" //avg or min


using namespace std;

/* Our domain of ascii characters */
static const char alphanum[] = "*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz";


class Map
{
    public:
        Map();
        Map(string seed);
        int width;
        int height;
        string values;
        char value_at(int x, int y);
        void set_value_at(int x, int y, char val);
        bitset<24> observe(int x, int y);
};

Map::Map(string seed)
{
    string delim = ":";
    int start = 0U;
    int end = seed.find(delim);
    string rows = seed.substr(start, end - start);

    height = stoi(rows);

    start  = end + delim.length();
    end = seed.find(delim, start);
    string cols = seed.substr(start, end-start);

    width = stoi(cols);

    start  = end + delim.length();
    end = seed.find(delim, start);

    values = seed.substr(start, end);
}

void Map::set_value_at(int x, int y, char val)
{
    if (x < 0 || y < 0 || x >= height || y >= width){
        return;
    }
    values[x*width + y] = val;
    return;
}

char Map::value_at(int x, int y)
{
    if (x < 0 || y < 0 || x >= height || y >= width){
        char temp = '*';
        return temp;
    }
    return values.at(x*width + y);
}

bitset<24> Map::observe(int x, int y)
{
    /*
    *
    * There are 8 zones around the rat, four close zones and four far zones
    * The close zones are the squares in a given quadrant that are 1 away
    * (e.g. to my left, to my top-left, and above me are the one away squares in
    * the -x, -y  or NW quadrant)
    * The far zones are the squares from each quadrant that are 2 or 3 moves away
    * (e.g. the squares with distance (-3,0), (0, -3), (-3, -3) are the outer
    * corners of the NW quadrant.
    * The quadrants are always ordered NW then NE then SW then SE first
    * close and then far.
    * This method outputs 24 0/1 values, first the 8 zones are asked if they have
    * an obstacle "*" those 8 answers will be senses[0] through senses[7] in the
    * above mentioned order
    * Then each of the 8 zones is asked is they have food, "$", again 8 answers.
    * Finally each zone is asked if it contains a pit "X".
    * These will act as the rat's "senses".
    */
    string targets = "*$X";
    string close[4] = {"","","",""};
    string far[4] = {"","","",""};
    int quadrant[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
    for (int dx = 0; dx < 4; dx++)
    {
        for (int dy = 0; dy < 4; dy++)
        {
            for (int qi = 0; qi < 4; qi++)
            {
                int sign_x = quadrant[qi][0];
                int sign_y = quadrant[qi][1];
                if (dx == 0 && dy == 0){
                    break;
                } else if (dx > 1 || dy > 1){
                    far[qi] += value_at(x + sign_x*dx, y + sign_y*dy);
                } else {
                    close[qi] += value_at(x + sign_x*dx, y + sign_y*dy);
                }
            }
        }
    }
    int counter = 0;
    bitset<24> senses;
    bool bit_value = 0;
    size_t found;
    for (int i = 0; i < 3; i++)
    {
        char target = targets.at(i);
        for (int qi = 0; qi < 4; qi++)
        {
            bit_value = 0;
            found = close[qi].find(target);
            if (string::npos != found)
            {
                bit_value = 1;
            }
            senses.set(counter, bit_value);
            counter++;
        }
        for (int qi = 0; qi < 4; qi++)
        {
            bit_value = 0;
            found = far[qi].find(target);
            if (string::npos != found)
            {
                bit_value = 1;
            }
            senses.set(counter, bit_value);
            counter++;
        }
    }
    return senses;
}

class NeuralNet
{
    public:
        NeuralNet();
        void setWeights(vector<double> weights);
        static vector<double> translateGenome(string genome);
        static string translateWeights(vector<double> weights);
        static const int n_i = 25;
        static const int n_h = 10;
        static const int n_o = 4;
        bitset<n_o> makeChoices(bitset<n_i> inputs);
        double W_ih[n_i][n_h];
        double W_ho[n_h][n_o];
};

NeuralNet::NeuralNet(){};

void NeuralNet::setWeights(vector<double> weights)
{
    /*
    *  Takes in n_i*n_h + n_h*n_o double values where n_i is the number of input
    * neurons (25 in our case the first is "pain" (did the rat just hit an obstacle?)
    * the other 24 are explained above in Map::observe)
    * The n_h (10 by my settings) hidden neurons are there for complex behavior,
    * they can kind of be thought of as emotional states.
    * The n_o outputs will be an encoding of the rat's next movement choice.
    */
    int pos = 0;
    int tpos = 0;
    int W_ih_size = n_i*n_h;
    int x, y;
    for (vector<double>::iterator it = weights.begin(); it != weights.end(); ++it){
        if (pos < W_ih_size){
            x = int(pos/n_h);
            y = int(pos % n_h);
            W_ih[x][y] = *it;
            pos++;
        } else {
            tpos = pos - W_ih_size;
            x = int(tpos / n_o);
            y = int(tpos % n_o);
            W_ho[x][y] = *it;
            pos++;
        }
    }
}

bitset<4> NeuralNet::makeChoices(bitset<25> inputs)
{
    bitset<n_h> hidden;
    for (int i = 0; i < n_h; i++)
    {
        double res = 0.0;
        for (int j = 0; j < n_i; j++)
        {
            double tval = double(inputs[j])*double(W_ih[j][i]);
            res += tval;
        }
        /* After checking which weights from inputs affect a given hidden neuron
        * If there is more positive than negative we fire it.
        */
        hidden[i] = res > 0;
    }
    bitset<n_o> output;
    for (int i = 0; i < n_o; i++)
    {
        double res = 0.0;
        for (int j = 0; j < n_h; j++)
        {
            double tval = double(hidden[j])*double(W_ho[j][i]);
            res += tval;
        }
        output[i] = res > 0;
    }
    return output;
}

vector<double> NeuralNet::translateGenome(string genome)
{
    vector<double> temp(0);
    for (int i = 0; i < genome.length(); i++)
    {
        char c = genome.at(i);
        double temp_val = double(c) - 82.0;
        temp.push_back(temp_val/40.0);
    }
    return temp;
}

string NeuralNet::translateWeights(vector<double> weights){
    string answer = "";
    int pos = 0;
    double temp = 0.0;
    for (vector<double>::iterator it = weights.begin(); it != weights.end(); ++it){
        temp = *it;
        temp *= 40;
        temp += 82;
        temp = round(temp);
        answer += char(temp);
    }
    return answer;
}

class Rat
{
    public:
        Rat(int start_x, int start_y, string genome);
        int isDead();
        void changeEnergy(int delta);
        bitset<24> observeWorld(Map world);
        bitset<4> makeChoices(bitset<24> senses);
        void enactChoices(bitset<4> choices);
        int x, y;
        bool hit_obstacle;
        int speed_y, speed_x, energy;
    private:
        string genome;
        NeuralNet brain;
};

Rat::Rat(int start_x, int start_y, string genome)
{
    x = start_x;
    y = start_y;
    speed_x = 1;
    speed_y = 1;
    energy = 30;
    hit_obstacle = 0;
    genome = genome;
    brain.setWeights(NeuralNet::translateGenome(genome));
}

int Rat::isDead()
{
    if (energy <= 0){
        return 1;
    } else {
        return 0;
    }
}

void Rat::changeEnergy(int delta)
{
    energy = energy + delta;
}

void Rat::enactChoices(bitset<4> choices)
{
    speed_x = choices[0] - choices[1];
    speed_y = choices[2] - choices[3];
}

bitset<24> Rat::observeWorld(Map world)
{
    bitset<24> observations = world.observe(x, y);
    return observations;
}

bitset<4> Rat::makeChoices(bitset<24> observations)
{
    bitset<25> pain_and_observations;
    for (int i = 1; i < 25; i++)
    {
        pain_and_observations[i] = observations[i-1];
    }
    pain_and_observations[0] = hit_obstacle;
    bitset<4> choices = brain.makeChoices(pain_and_observations);

    /* To see input observations and output decisions uncomment the below code
        cout << "observed "<< pain_and_observations.template to_string<char,
         std::char_traits<char>,
         std::allocator<char> >() << endl;
        cout << "chose to do: "<< choices.template to_string<char,
         std::char_traits<char>,
         std::allocator<char> >() << endl;*/
        /* choices[0] is the choice to move down by 1, choices[1] is to move up by 1,
        *  choices[2] is to move right by 1, choices[3] is to move left by 1
        * if all four are 1 then the rat doesn't move for example, while if it is
        * choices[0] = 1, choices[1] = 0, choices[2] = 0 and choices[3] = 1 then it moves
        * down and left */
    return choices;
}

int simulator(string mapseed, string genome, int start_x, int start_y)
{
    Map board(mapseed);
    Rat arat(start_x, start_y, genome);
    int cur_x, cur_y;
    int target_x, target_y;
    int dx, dy;
    char next_spot;
    int moves = 0;
    while (!arat.isDead())
    {
        moves++;
        cur_x = arat.x;
        cur_y = arat.y;
        /* cout << "rat at "<< cur_x << " by " << cur_y << " on move " << moves << endl; */
        arat.enactChoices(arat.makeChoices(arat.observeWorld(board)));
        target_x = cur_x + arat.speed_x;
        target_y = cur_y + arat.speed_y;
        next_spot = board.value_at(target_x, target_y);
        arat.hit_obstacle = 0;
        if (char(next_spot) == char('$'))
        {
            arat.changeEnergy(10);
            board.set_value_at(target_x, target_y, '.');
        } else if (char(next_spot) == char('X'))
        {
            arat.energy = 0;
        } else if (char(next_spot) == char('*'))
        {
            arat.hit_obstacle = 1;
            arat.changeEnergy(-10);
            target_x = cur_x;
            target_y = cur_y;
        }
        arat.changeEnergy(-1);
        arat.x = target_x;
        arat.y = target_y;
    }
    return moves;
}

/* NOT ANDY'S CODE ANYMORE AFTER THIS LINE!!! */

/* -----------------------------------------------------------------------
 * getRandCharFromGeneDomain
 *
 * Gets a random character from our alphanum ascii domain of possible gene
 * characters.
 * -----------------------------------------------------------------------
 */
char getRandCharFromGeneDomain()
{
    return alphanum[rand() % (sizeof(alphanum) - 1)];
}

/* -----------------------------------------------------------------------
 * Gene
 * string genome
 * int fitness
 *
 * A gene with a fitness level and a genome.
 * -----------------------------------------------------------------------
 */
class Gene
{
    public:
        string genome;
        int fitness;
        void setValues(string g, int f);
        void setFitness(int fit);
};

void Gene::setValues(string g, int f)
{
    genome = g;
    fitness = f;
}

void Gene::setFitness(int fit)
{
    fitness = fit;
}



/* -----------------------------------------------------------------------
 * makeRandomGene():
 *
 * Generate a random CHROMOSOME_LENGTH char ASCII string, that will represent our gene.
 * FORMAT: The format of your genome will be a CHROMOSOME_LENGTH length ascii string
 * with ascii values between '*' and 'z' (42 to 122).
 * Credits: http://stackoverflow.com/a/440240/4187277
 * -----------------------------------------------------------------------
 */
string makeRandomGene()
{
    char randGene[CHROMOSOME_LENGTH + 1];
    int randNum;

    for (int i = 0; i < CHROMOSOME_LENGTH; i++)
    {
        randGene[i] = getRandCharFromGeneDomain();
    }

    /* Null terminate the string */
    randGene[CHROMOSOME_LENGTH] = 0;
    return randGene;
}

vector<Gene> seedGenes(){
	cout << "seeding genes..." << endl;
    vector<Gene> seeds;
    for (int i = 0; i < 35; i++) // should be one more then max number of seeded rats
    {
        Gene g;
        g.setValues(makeRandomGene(), 0);
        seeds.push_back(g);
    }
    seeds[0].setValues("pJGWQ4joKruUmc2InY`tXx-+k^Eta-4f>iG+3AzTBL/Dy7PlvqKSF1VGmMkcuL_bEw\\CQ+?veB?c_y0ZWp3D>;O@^O90qnx1zZZMJbH4u^,w8gKHWRj0svVfJe+QXDXXiG:8*iC4\\UtMBB03.PI1Ku,+p@CM,?hk\\[9lIb6*<?4TWImf5KmVUou=1L-AqM4b-<ds:O\\]B>Fqq8\\zYhcCXmdY>gsy9]+MveTZj<xtuDzlb[zdNsi5.Vu8NqdiQfH4v5l*]d[tOjY5Yfwrz0+n>]nJ@gMgb@On=@",0);
    seeds[1].setValues("wNlV.DxsrleBq>u\\SI9HL>-bz=xYa?B5@BO9Vg4<b^_AmV_w/Ev/01LfDncUBKOQ_yA7PfiUhd2D7CPsAp;TX/ajM3G*z?xOOBv,QQfwh\\7KzgP/<@Gq\\4kuhrLmRy8N,VBNa3rhG_phE6,hR.i*CwvAVaQL_^EO7BZdvjl4<_A^BRp?hQiezjp.jvxdVgYOh3z;3*FJG+e?H6lEo+d/F,W2udS7CEp;xE^G?EHb7ouAA:JR*BK1AfTNLFX;+,v[b@TkKL-oS]If`Tw;q<[P1`kH?tLjIwe=`-", 0);
    seeds[2].setValues("wNlVdDxsrTB+8>u\\\\I9HL>-bz=xYa?B5@Bd9VgJ<b^_A,V_w/Ev/01LfDnH>BKOg_yA7PfiUhd2D7CPsAB;TX/ajM3G*z?xOOBv,QQfwh\\7KzgP/<@Gq\\4kuhrLmRy8N,VBNa3rhG_phE6,hR.i*CwvAVaQL_^EO7BZdvjl4<_A^BR.?BQiesjp.jnxdVgYOh0z;3*FJGye?H6lEo+d/F,W2sdS7Czp;xE^G?EHb7iuAA:JR*KK1AfTNLFX;+,s[b@TkKL-ow]If`Tg;q<[P1`kH?tLjIwe=`-", 0);
    seeds[3].setValues("tTv:aV<4fmHOzsXs_-,geDc]8Gwoe7FkqK\\*+?4@siw;aS3@fE5Sb\\?@<cb=l515LnPV;JBEi=x;f\\l.^AgF7<mIITavoy1[vcT5<:mhesy[UzBlEUf,4hChCgdFe*lrg-13Pnh/f/w5>bpi=kPr4Vo.ShZ=h\\9e_p-4D\\v?8Pv]LkK`krgoM[X5Y<IFnh@j>C+X]8S5byPdl.PNVq2XYhdtOBO,?Rt]Nu=41D21BJLKxu1=a3M;8Z5U1jZPEc?8^WOKqV\\I?toF5tTkCK8a:S>QB--mX8rI..", 0);
    seeds[4].setValues("BJLfa@Igkl=Pf^ij?B?xQPwRBfw-/V,cv8P[eKzh*sd7+pE9fql9z_/fqNd-4?@Yij[O,CR6.wR@+61x0HtO,KQSw`x;vR,_GG3zxa5eRC*Tm?`_m+[oL]AYAOzK];2fXuhNxA8_;8GDOK/ihj-+FDAW?NVL7m`zX/@Sv,laH1mWds4PUj4=F@IhWf]I>KJ41M^mOC[7,T7fMTDqaAm]j?rZ9Kvo5<la9TV[IYaB=WzDkF,<sCfrh?XxHgddQB4tAk3gXj=+WLG:e=UaV2ESKqlDe9[jz.0S>H", 0);
    seeds[5].setValues("Sfz2smu:zej,9j:R\\[thR73RzRdP0U.v2TF9-X/PTlOPdYF+k1N;o3`[uTfT>RTq:rsTBk:gYlNq-[@OmMrvVo^U\\z=JV>l.tAq*\\BLM3vJQLfnnpr_mN-7RN;^.-9j4ddZKXN,gvH;x:qYAgPz+tb29Li[qJ/jH+UepmPpV0qC^u2tpvl_/Z-`OTmijR@XW5iJRk0/\\ztnzDHBu+HmsY*,AlZy`aJZ?WFQf*?>3`z6\\TiU+O,MvIH\\7zL?:=/.w^V/XH-`[X+8L7.+zxDBEm;jwW29jp/oj<Y",0);
    seeds[6].setValues("wNlVdDxsrTB+i>u\\\\I9HL>-bz=xYa?B5@Bd9VgJ<b^_A,V_w/Ev/01Lf+nH>BKOg_yA7PfiUhd2D7CPsAB;TX/ajM3G*z?xOOBv,QQfwh\\7KzgP/<@Gq\\>kuhrLmRy8N,VBNa3rhG_phE6,hV.i*CwvAVaQM_^EO7BZdvjl4<_A^BR.?BQiesjp.jnxdVgYOh0zC3*FJGye?H6lEo+d/F,W2sdS7C,p;xE^G?EHb7cuAA:JR*KKAAfTNLFX;+,s[b@TkKL-ow]If`Tg;q<[P1`kH?tLjIwe=Y-", 0);
    seeds[7].setValues("wNGV7DxsrTB+i>;\\\\I9HL>-az=xYa?B1@Bu9Vgc<bw@A8V_w@Ev/01Lf+n0[BK^g_yA7PfiThd2D7CPsAB;TX/a`M3G*z?xROBv,QQ9wh\\7KzgPb<@3q\\>kuhrLmgy8N,VBNa3rhG_phE6,hV.i*CwvAVaQM_zEO7Bbdvjl4<_AJBR.?BQiehjp.jnudVgY5h0zA3oFJ*ye?H6QEo-d/.,W2sdS7C7p;xE^L?EHb7cVAA:JR*KKAAfT>LFX6+2sf@+TkKL-uwJIf`Tg;q<[R1`kH4tLjIwe=Y-",0); //best
    seeds[8].setValues("+PQEHC9BYV\\ky2AO<Z;=VFKWUPPXUY`RWLXeoa:VFEU[;BYILP_V[E@[NRpQ96bXIZE=`>\\CrK-^ZA0Zq]3O6Hb?@AMPQkHFlMKEVI=NVfS_LGQ:82@C8qRCpKEQHT;ZLxXWAGI\\\\=w=]YGi9\\LL43nJTLH:K?A>Q<,A0GHJ;9oP,<_WM@A;p[]EPKA]HMwdPL[I];Gb`P:pR*WdRRC4<\\`[\\ZNCU9GN9Q:LFZX8LQyOYOJ4;pBV;UJ-<8bMPXgM7dOMRH<qhlQZUHJId<8iJtAi5[SE]0URBY", 0); //best opponent
    seeds[9].setValues(",SS2;++IzNNNq\\\\\\k2222ssrUU?8yIffr-qsMMMccnnnn?QcQ*jjj^hcceece??:R@h;UCCC---FFss>>ffLYYhhYB,--LddbbbL5iiiNNN*Ly]]=B_rrppHyya;Nc55FkkkUoo-x/-->JJ77@`qSSAMevCC_fvs?..>lAA---TTT44pp]Y]]f\\diiATo>ToA4KK222I.I99<<<1TTuuTNNmm888::000Aq4*.<<vk,8,8,gg?Q\\\\\\\\jqqq@@*m@NNJ<<K_Nm/QQZZBBBBJaLaaEEqhj``jjb,", 0); //next best opponent
    seeds[10].setValues("e;:pjzM^YR83fs5e^'cAadENn'LJDW.dtjFif;8fzD:u0_>F6QeE*L+btnGu/BUL[bf5/J@Ymfo>=q_E7vzNgiKa8I2Sj8mFyb[7KT61IOvg=Y*^hq_iX:=D^>17qm-i3;j@3T2j[,Id<Vo3dq/MQr_d^D6WlCPb7j=5O?MVKhLNV/meLlqikdJ@,=vWM3WI1Ln-CxyU:'gXN7Xci*udaYCm'GoUoi^gNsobMPx3J<fjeHDI:@>l:_I'1ku/wHgL/M4w*jwsgNPKvAC=6PThMp]FU[;dcPjSzF", 0); // brute force
    seeds[11].setValues("RRRRRRRRRRRRRRRrR2RRRRRRR2RrRRRRRRRrR2RRRRRRR2RrRRRVRNRVRNRRRVRNRNRVRRRNRVRVRNRRRNRVRNRVRR.>vfR2RvRR.>vfRbR2RRvf.>R2RvRRvf.>RbR2RRRFR^RFR^RRRFR^R^RFRRR^RFRFR^RRR^RFR^RFRRzz**zz**RRzz****zzRR**zzzz**RR**zz**zzRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRzRRRZRRRRzRRRZRRRRzRRRZRRRRzRRRZRRRRRRRR", 0); //hand crafted
    seeds[12].setValues("denO*PJ.]7\\\\qaN2NY2arUsUY8sIyU-qrMfMfcMsnnnQccn*QQq^jAjejcgc:R?he;?CU@---FCFC>sFLYfh>hfBY6Ld*b,b-LbdiNie5*iyLNB_=r]p=Hpr;Na5y5yk5cookGk/k---77J`>qJSS@vCe_AfMsvC>A.A?-.--A44tpT]T]YpdifA]Af>oi4KA2>2oI2K9<9<]1ITT<NBTmu8u88m000q:4:D*Ak,1,<8bgJ8\\\\Q\\gj?qq\\_Nm/QwZZN<<<DaLaq@@*aEEqm@NNjjb,h3`DBBBB", 0); //hand crafted
    seeds[13].setValues("qKJgCX7y0^D:M9/qIuN0CB4e-]nPb8z=n+]V/9snyN_al]U;8.?YN>YN]F_+eb0Ds\\GO@cY+r7@qQt\\50Udn@5twzI^3d>Fzt*bhfH.@u]nbFU+?-[0ehggcmMK16arZodq,YvWfAzUOcpi5GZpw\\LfxAT/z:/3LIUK>LQ=XtOQgYsB2O5HIZ.ui^]=LgM05e@T]*_,uEqeLQ:wY,@_2;UyQrQB<zz3PHJIyCe\\\\5Vv9Kz8D+DR/WaR/*</Uu??uz*v;Hk:^C0aBtgLZDScSg7zIVw@TEF\\U/>d", 0); //robust rat
    seeds[14].setValues("]JEd>Qu4[E.FM>1q4lc?Y,7_5OlhXsmv<ZXQ]c8MnoEhB\\,c]RC<x`0W5Wqxsk;qvonVMyk>>lgw/:Pt+z8K@euE.xF9<X<k\\3C.M.RP7D1_lHYt@<EV]/S_:A,1.>npS1t6EK\\RRM3ZuNF:F,wjQqpbGs=[FcP/.YWc:5Jr_SQY6FjRLUas+WjI_F/*;p?=k@J3a9paXqD+b`x+aNrpeGivl4abCu45ptNVNfzr_82V40W*5^0IBsDDYRDswDrNObWiY.pAo8xJN3`7M>fuCL=rMMlLw=MK4z", 0); // 200 level rat
    seeds[15].setValues(",W?1U@JT^9v2qNyJkrFF=QXMF_F3g+t^,3TdaJzxx]Ejt\\]QOK>cyl5l_w,]NN+Faw.Zf4xSGBvO3u8YJL^HN\\9LChgCjx2`td`Ox]N,4j==>[+q<bm`TozHyvmJs+]+uDv24_eNu@P1qsEBkS8D:96>CWK;cD=nfSKW\\Du,L@HhuS8fe;u@cw1Q0Q0v[Dhim<46*D`Sj\\-._4]J`VI>bunT2fPrxAyIcHeCd76ceVI1+*a?fWOMa97j*:XN2fz*vD`ofCW-M1UjEM3wze\\`7w_-aUDj5iQ;R,", 0);  // 180 level rat
    seeds[16].setValues("59oLFIScY=cHnVJm/wVaZ43AJRbYfhdE_meOvq8z`k`SsWEwo@-q>/I.**MfSG<28wZF9O\\oU,X]6Bz=Y,B,Qb=L?Cr3nU>HD,5Ub1463bbdzNjs1+F[sanB:5jBrRAqIpn06xR5_Wu-lmW2nv52RrHbwpw-;Z[lB^/D[b2;TbSj,J+t/zD]+dUf0CIb]_Eh=I_gHgs-z/[/GY8`.,RnU,?^cg?uUgUjwQq*rT2MUOW?5pf,2Lh]_p5GGOzujlsLCtug]dm7>1Om6Tx>x4JGUJKG`HAG9IyRRM", 0); // 190 level rat
    seeds[17].setValues("VkNU[\\JkpJNJ\\7xnMrd=il.F[VaZ@DngCRQD3r*8Qh0Bvf6X]pkYq.uQZl=qUCL>kDitKs2rqq:+lVY_[Z=Q^77>yM4Sfwxf@foan6id<ye.dTcD3v*gB7*QjD9eA6PgrUNfqL_3JJ7C3/]<:myRz9yzSHdjTJgLuJGkl+3K[PddUVvuH4]GCkV+8@+c`hDj7qjyr3IcYBLDnXN;r0hJrDK/jLx_IQ^fGMdOfB7Ejio]VB-MIlmPENU4/SiO9L:W.4+k\\HELFI>saBU?CHe^l@x0yfU7Gvt[?4", 0); //all $
    seeds[18].setValues("1LNU[sJ5phM_^7BmM4d=il.F[wMZMDLgCRzD;r*8Eh0oC4Hw]pkYqcuQZlMqUC6mkGitTs2rqq:+lVY@[Z=;^67>xO+Sfwxf@foan6il<yk.qTPJ3v*lBE/QjD<@A6Pgr9MfhL_3JJ763/]<:myRz9QzSRdjTJlLuJGkl+3KZPh@U;?uHB]GCJ<+*@+c`hn:7qjyrMucY77DnXh;r0hJrD;/jLx_IQ^/GMmOlB7Eji>j=B-MIl_PEzB4aSiO9L:g.4+k\\HELFI>saBU>CHe^l@x0yfU7Gvq[?4",0); //all $ even better
    seeds[19].setValues("1GNU[s\\PphM_^7vmM4M=il.F[w,ZMDLgC-zD;r*8Eh0oC6Hw]pkYqcuQZlWqUC6mkGitTs2rqq:+lV_@[Z=I^67>lM+Sfwxf@foan6il<yk.qTPJ3v*lBEMQjD<LA6Pgr9MfhL_3JJsE3/]<<myRz9QzSRdjTJlLuJGkl+3KZPh@U7kuHB]GCJ<+N@+c`hk:7qjyrMXcY47DnXd;r1hJrD;/jLxoIQ^/GMmOoBTEji>j=B-MIp_]EzB=aSiO9L:g.4+k\\HELFI>saBU>CHe^l@x0yfU7Gvq[?4",0); //all $ even even better
    seeds[20].setValues("-aNV2DOhJhMf^7vmM=M=wz1P[w,hMDLgB-FD;r*8Eh0oC6Hw]pkYqcuQZlWPUG6tkFitTs2rqqA+lV:@[Z=I^67>lM+Sflxf7foan6il<yk.uTPJ2v*lBEMQjD<s96Pgr9MfhL_3JJnE3/]<<myRz9QzSNdjTJlLuJGkl+3KZPh@U7kuHF]GCJ<+7@5c`h7*7qjyrM_cGt7D3Xd;r1hJSP;/jLxo;Q^/G1FOjBBEjM>X=B-qIp_\\+z3=aSiO9L5g,4+m\\HDLFI>saB[>CHe^l@x0yfU7GvqM?B", 0); //all $ really good
    seeds[21].setValues("-aNV2DOhJhMf^JvmM=M=vzLP[w0hMDLgB-FD;r*8Eh0oC6Hw]pkYqcuQZlWPUG6tkFitTs2rqqA+lV:@[Z=I^67>lM+Sflxf7foan6iz<yk.yTPJ2v*lBEMQjD<s96Pg99MfhL_3JJnE3/]<<myRf9QzSNdjTJlLuJGkl+3KZPh@UAkuHFkGCJ<+7@5c`h**7qjyrM_cGtFD3Xd;r1qJwPm/jLxoKQ^/G1.OjBBEjM>X=B-AIp_h+z3=aSiO9L5g,4+m\\HDLFI>saB[>CHe^l@x0yfU7GvqU?B", 0); // all $ slightly better
    seeds[22].setValues("?y7V2DOhJDif0JvMMPC\\vnLP[w0bMDLgB-F5;r*8Eh0oC0>w]pkYqcuQZmuPUG6-kFitTs2rqqA+lV:@[Z=I^67>lM+Sflxf7foan6iz<yk.yTPJ2v*lBEQQjD<s96Pg9/MfhL_3JJ[E3/_<<myRK9QzSNdXTJlLuJGkl+3KZPD@[AkuHFkGCJ<+7@5c`h*77qjyQM_^Xt+D3X];rsPuf>U_c>xojQ^/G1.OjABEjM>X=B-AIp_h+z3=aSiO9L5g,4+m\\HDLFI>saB[>CMe^l@x0yfU7GvqU?B", 0); // all $ slightly slightly better
    seeds[23].setValues("?y7V2DOhJDif0JsM/PC\\vnLP[w0bMDLgB-F5;r*8Eh0oC0>w]pkYqcuQZm]PUG6-kFitTs2rqqA+IV:7[Z=I^67>lM+Sflxf7foan6iz<yk.yTPJ2v*lBEQQjD<s96Pg9/MfhL_3JJ[E3/_<<myRK9QzSNdXTJlLuJGkl+3KZPD@[AMuBFkQCJr+7R5c`h*77q^yQM_^Xt+D3XL;rsP*f>U_7>xojQ^/31.OBABEjM>v=B-AIp_h+z3/aSiO9L5g,4+m\\HDLFI>saB[>CMe^l@x0yfU7GvqU?B", 0); // all $ 4200
    seeds[24].setValues("?y7V2DOhEDif0Js]=P*\\vnLP[wBWMDLg]-Fc;r*8Eh0oC0>wzpkYqc\\QZm]PUG6-kFitTs2rqqA+IV:7[Z=I^67>lM+Sflxf7foan6iz<yk.yTPJ2v*lBEQQjD<s96Pg9/MfhL_3JJ[E3/_<<myRK9QzSNdXTJlLuJGkl+3KZPD@[AMuBFkQCJ9+7n5c`h*77q^yQM_^Xt+D3XL;rUj*f^U_7>xojQ^/31.OBABCjM>v=B-Aip_h+-3+aSiO9L5g,4+m\\HDLFI>saB[>CMe^l@x0yfU7GvqU?B",0); // all $ 4230
    seeds[25].setValues("?i7V2AOhECif0Js]=P*\\vnLP[wBWwFLgxR=c;r*8Eh0oC0>wzpkYqc\\QZm]PUG6-kFit`s2rqsA+IV:7[Z=I^67>lM+Sflxf7foan6ij<yk.yTPJ2v*lBEQ5jD<s96Pg9/MfhL_3JJ[E3/_<<myRK9QzSNdXTJlLuJGkl+3KZ_D@[A5uBFk5C4[+7n2c`h*77q^_TM_^Xt+D3XL;rUj*f^U_7>xojQ^Y31.MBABC]M>v=F-xip_h+-3+aSiO9L5g,4+m\\HDLFI>saB[>CMe^l@x0yfU7GvqU?B", 0); // all $ 4260
    seeds[26].setValues("?i7V2AOFECif0Js]=P*XvnLP[wBWwFWgxR=c;r*8Eh0oC0>wzpkYqccQZmIPUG6-kFit`s2rqsA*IV:7[Z=I^67>lM+Sflxf7fhap6ij<yk.yTPJ2v*l?E[/jD<s96Pg9/MfhL_3JJ[E3/_*<myRK9QzSNdXTJkLuJGkl+3KZ_DM[A5;BFk534[+7n2c``777q^u[M_^Xt+[9CL;rqD*f^m_7xxoRQ^Y31.MBABC]M>v=FJrip_h+-3+aSiO9L5g,4+m^HDLFI>saB[>CMe^l@x0yfU7GvqU?B", 0); // all $ 4270
    seeds[27].setValues("?i7V2APkECif0Js]=[*XvnLP[wBW6FWgFR=c;r>8Eh0oC0>wzpkYqccQZmIPUG6-kFit`s2rqsA*IV:7[Z=I^67><M+Sflxf7fhap6ij<yk.yTPJ2v*l?E[@jD<s96Pg9/MfhL_3JJ[E3/_*<myRK9QzSNdXTJkLuJGkl+3KZ_DM[J5;@Fk53p=+7n2c``77Uq^uSMt^mt+]9CLA>qD*j^2_7xxoRQ^531:MBAyC]M`v=FJrid_hx-3+aSiO9L5g,4+m^HDLFI>saB[>CMe^l@x0yfU7GvqU?B", 0); // all $ 4280
    seeds[28].setValues("S58DQ<R1:dpzMpv>._Yj0\\i?UXXTs2z\\NGjV0b^2/h@w]q;9]r^0Ewgof6GUY`1D>.JVVNQ6xKQnQNCIKC>r-.T9e2MSW?VR7I.SRd/OGMgk2A:p>r;z<?`wDBC-[R[xGYrN*VJhY+]o6m?Z:@0E^M;;.DCrrilFdbE/lLK[FT[jVXIfXOP;;OxKYHD;JF15P2,SPSCHFW7HOHC/uqd/8-2P=Ht.:cCz0u=5Os9lL==B3qjEP[tYROX_cB8Io>[mSsVp+40K*m-nyfmpe+nNeg8^c<C3J<b30[",0); // all $ 2080
    seeds[29].setValues("S_i5uCJ4OR+[npA9M7+.dpOz4JqD4t6+IUvUuvauhajDh^:Okr`aQRaogfP:bSf\\?:FO];[r]xb:oLcbDqJ?39T;Uh.X;0cP@0z\\BlpZhUu`6V^_7t401W6;MfVoK.Ta4Y[\\J4Kep8:;*MqGm1k\\/eM6.bVN*6S4y*G>DGLIRl]lQ*8S0<j5Dv[Es-UKITyC7^T909[,ezFpg/4S,lx2v`J1+k[?K-:?]2=tXEcwkOn*GBXUudpD26Igh^v8\\@vNHJO_CRF3[]_ugMYu\\c0bjBwbz;YR2T@J3u",0); //all $ 4000
    seeds[30].setValues("r>HFW<ITO<\\g.AMox-7\\UoxXvX.U@U;cJ6,2jLpknYeN1.>5,iwbIrfT]2=[GUhfyLblAQuHOoCQ_SrcQo`/oNJ:<+Kooq7,ucF;22ZpD2XG[jo:VQGu;FY9G3LKLW^sABF-@mrj0RLh:UgfDKrU`>J<Hp^j^AE1@LEf6<V2u7nu7*L,4U]Cj6[R0L;+O:L5^LMp=j>EkpU_W+AMWovk+RpM/JD6X67RAD>o>3;DnXYki^tHn;qmKN>5R@@G7kpj=y9ZZd1bXAXFIor<4^cOgSmYiAw09:gUq>", 0); //all $ 4650
    seeds[31].setValues("5Nztrtl*cU]N^2D/ra86DUbS5<zrxli4Otgvn73NdgTh+KAF]m.TYw9gVp]Pc[jXn>3/WL7P8?Mz[ez,WjV,PAYP3UtpBc,DjVej^g2vSf50efmL`Ta;yb84-7zQ?,1dG_iD[6ZMl_HaaU2VA?G+EqGYyF4.Y9J6-8?nOu@@Unx8W/_iI1mG<q==*Bw6aRWenFK[a.8KrKc[iz3XSw7otLyx\\lAC@8-ZR?5]7bG6<Xrk>x4:hX[gky,MAjf?YzV4*]fA\\iKnm-:,YD;rW;9W[az;P2+VpmIuX4", 0); //all $ 3870
    seeds[32].setValues("`troemBjY+q8843gt7l\\94Oo.;6>ID0c;jTLND04bytkF+K/hFF*pk9t@blmZZ7u@1m,a]<h7x0Deo0ZepUmy-MNqS6o1bVW_I8Ve>:kv_ZVl/CZEO]4Ovu6Ua+l=?\\`nG91E.T\\19b]O4vuiER/afSlGT@4+;rdO@bk`t6wm-jg`UoFEN;-:HJnMqODjF@g_*\\NZvQ]e-Mf:In:u[@KFQxIh=^aSk,>iV`nKcMX`SrNxC?dVpyqEY4FlGv9k9RdwQr1=j2,40PIM[Y/`*KxCKoKp3Zqizkus3",0); //all $ 4780
    seeds[33].setValues("nrN.eu-<6_V884W8HGxO34vol:6DI40c;rTSND44EytkF+d/hF7*pd9u@bGmZZ7.@1V,a]<h9irDMotZepUmk-oNoS6o3jVWbItUe>Akv_ZV^9SZE]]4Ovu/Ua+p=?6`cG91E[T\\19H]O0vuiER/afSlIK@4=;rdO@bk`t6ws-jgkfoFENR-:HNnWq>]jF@g_avNr1Q]B-9fofW:u[@KGQx^h=Kaad,->VM=Mchc`6;NxCii_m2qEYZplGp9k9Rd_Qr1=j2,40PIM[Y/`*KhaxoKf3Zsizkus@", 0); //all $ 5200
    seeds[34].setValues("nr_B8u;xs*V884W@=GxO34vod:6DK93c;ATSyD44Eyt_Kld/jV7*pd9u@b,mZ\\7.@1c,a]<hGPrD@otZepUMk-mNoS6o3yVWbItUe@A6v_ZVy9SZElX4Nv\\/Ua+p=?6`cG91E[T\\19H]C0vuiER/-fSlI]@4=;fdO@bk`t6ws-*gk?oF+@R->HKnWq3]cFi;_aJNnGQ]BP9fopp:B[@KGQmnhg4WX9b[>Lo=1ctck6zlxC9w_m2q_GZ[l[p9k9Rd_Or1=j2,40PIM[Y*`*Khg>oKf3Zsizkus@", 0); //all $ 5210
    seeds[35].setValues("nr_BU-uxs.V883W@=GxO34tod:6D>93c;PTSJD343ythKlT/jV<*pk9u>b,mZ\\7.@1c,a]\\hGCwD@o^ZepUMo-mNoK6o3ytW/IAUe@A6v_OVy9SZElX4Nvd/Ua+p=?F`cG91E^T\\19]]F0vliER/rfSlI]@4,;fdO@bk`t6wp-*g5?oF+@R4VHuQWq3]c/B;6aJNnml].P9=wpp:Bc_QHcPnQd4WX8Fd<LU=1=tck6zlxN;w_h2qPGt[l[p9k?Rd_Or1=j2-40PIM[Y*l*Krg>oKf3Zsizkue@",0); // $ 5230
    cout << "done seeding" << endl;
    return seeds;
}

/* -----------------------------------------------------------------------
 * makeGeneVector():
 *
 * Returns a vector of n randomly generated genes of length CHROMOSOME_LENGTH. Gives each
 * object in the vector a fitness level of 0 to start it out.
 * -----------------------------------------------------------------------
 */
vector<Gene> makeGeneVector(int n)
{
    vector<Gene> geneVector;

    /* Make INITIAL_POPULATION amount of random genes and push it to firstNGenes */
    for (int i = 0; i < n; i++)
    {
        Gene g;
        g.setValues(makeRandomGene(), 0);
        geneVector.push_back(g);
    }
    return geneVector;
}

/* -----------------------------------------------------------------------
 * runPopThroughMaze():
 *
 * Given sim parameters and a vector<Gene> population of randomly generated
 * rats, run them through the maze and fill up each Gene object in the vector
 * with a fitness level. Fitness Level is the same as Num Moves survived
 * -----------------------------------------------------------------------
 */
void runPopThroughMaze(string mapS, int startRow, int startCol, vector<Gene>& pop)
{
	int rows = sizeof startPOS / sizeof startPOS[0];
	int column = sizeof startPOS[0];
    for (int i = 0; i < pop.size(); i++)
    {
    	int moves = 0;
    	for (int j = 0; j < rows; j++){
    		int move = simulator(mapS, pop[i].genome, startPOS[j][0], startPOS[j][1]);
    		//cout << "rat: " << i << " maze: " << j << " moves: " << move << endl;
    		if (MULTI_MAZE_WEIGHT == "avg"){
    			moves += move;
    		}
    		else if (MULTI_MAZE_WEIGHT == "min"){
    			if (moves == 0 || moves > move){
    				moves = move;
    			}
    		}
    	}
    	//int moves = simulator(mapS, pop[i].genome, startRow, startCol);
    	if (MULTI_MAZE_WEIGHT == "avg"){
    		pop[i].setFitness(moves/rows);
    	}
    	else if (MULTI_MAZE_WEIGHT == "min"){
    		pop[i].setFitness(moves);
    	}
    }
}

/* -----------------------------------------------------------------------
 * rouletteSelect
 *
 * Given a vector of int weights - select one index for the vector with
 * a probability based on the weight. The higher the weight, the higher
 * the chance that index will get selected. NOTE: Must srand(time(NULL))
 * in main ONCE AND ONLY ONCE in the whole program.
 * -----------------------------------------------------------------------
 */
int rouletteSelect(vector<int> weights)
{
    int rouletteSize = 0;
    int roulette[weights.size()];

    for (int i = 0; i < weights.size(); i++)
    {
        rouletteSize += weights[i];
        roulette[i] = rouletteSize;
    }

    /* We roll the ball to see where we land on the roulette */
    int throwBall = rand() % rouletteSize;  /* Random number from 0 to rouletteSize */
    int i = 0;
    while (roulette[i] <= throwBall)
    {
        i++;        /* Ball moves to the next roulette slice */
    }

    return i; // return the index
}

/* -----------------------------------------------------------------------
 * chooseMates():
 *
 * Given a vector<Gene> which represents the population/gene pool, select
 * TWO mates with the Roulette Wheel selection algorithm. So basically each
 * gene is weighted based on its fitness level, and the higher the weight
 * the higher the chances it will be chosen to be mated.
 *
 * We return those two genes in a struct format.
 * -----------------------------------------------------------------------
 */
struct TwoGenes{
    Gene first;
    Gene second;
};

TwoGenes chooseMates(vector<Gene> & population)
{
    TwoGenes chosenMates;

    /* A vector of the fitnesses of our population vector */
    vector<int> populationFitness;

    int chosenMale = -1;
    int chosenFemale = -1;

    /* For each rat in given population, push their fitness
     * into our populationFitness vector so we can run roulette */
    for (int i = 0; i < population.size(); i++)
    {
        populationFitness.push_back(population[i].fitness);
    }

    /* First we select the male mate */
    chosenMale = rouletteSelect(populationFitness);
    chosenMates.first = population[chosenMale];
    population.erase(population.begin() + chosenMale);
    populationFitness.erase(populationFitness.begin() + chosenMale);

    /* Then we select the female mate */
    chosenFemale = rouletteSelect(populationFitness);
    chosenMates.second = population[chosenFemale];
    population.erase(population.begin() + chosenFemale);
    populationFitness.erase(populationFitness.begin() + chosenFemale);

    return chosenMates;
}

/* -----------------------------------------------------------------------
 * reproduce():
 *
 * Given a male and female gene, reproduce numChildren children. Then
 * return a vector of Genes of the children. Will introduce random rare
 * mutations.
 * -----------------------------------------------------------------------
 */
vector<Gene> reproduce(Gene male, Gene female, int numChildren)
{
    vector<Gene> result;
    string genome1 = male.genome;
    string genome2 = female.genome;
    for(int j = 0; j < numChildren; j++) {
        string output;
        int totalfitness = male.fitness + female.fitness;
        char c;
        for(int i = 0; i < CHROMOSOME_LENGTH; i++) {
            int randNum = rand() % totalfitness;     /* Generate a random number between 0 and totalfitness */
            if(randNum < (male.fitness / 2)) {
                c = genome1[i];
                output += c;
            }
            else if((randNum > (male.fitness / 2)) && (randNum > (male.fitness + (female.fitness / 2)))) {
                c = getRandCharFromGeneDomain();
                output += c;
            }
            else {
                c = genome2[i];
                output += c;
            }
        }
        Gene child;
        child.genome = output;
        result.push_back(child);
    }

    return result;
}

string mutate(string oldGene){
    string newGene = oldGene;
    int len = oldGene.length();
    int randomSpot;
    int mutateRoll;

    for (int i = 0; i < len; i++){
        mutateRoll = rand() % GENE_MUTATE_OUTOF;

        if (mutateRoll < PERCENT_OF_GENE_TO_MUTATE){
            randomSpot = rand() % CHROMOSOME_LENGTH;
            newGene[i] = getRandCharFromGeneDomain();
        }
    }

    return newGene;
}

void keithsReproduce(Gene male, Gene female, int numChildren, vector<Gene> & populationToAddChildren)
{
    int crossoverDiceRoll;
    int transferPointRoll;
    Gene mostFitParent;
    int totalfitness = male.fitness + female.fitness;

    /* Find the most fit parent */
    if (male.fitness > female.fitness){
        mostFitParent = male;
    } else {
        mostFitParent = female;
    }

    /* Reproduce! */
    for(int j = 0; j < numChildren; j++) {
        Gene child;

        crossoverDiceRoll = rand() % 100;   /* RandInt(0, 100) */

        if (crossoverDiceRoll < CROSSOVER_RATE_PERCENT){
            /* Now randomly find the point at which we are going to do crossing over */
            transferPointRoll = rand() % CHROMOSOME_LENGTH;

            /* Now we copypasta a chunk from male and chunk from female */
            string firstChunk = male.genome.substr(0, transferPointRoll);
            string secondChunk = female.genome.substr(transferPointRoll, CHROMOSOME_LENGTH);
            string newGenome = firstChunk + secondChunk;

            child.genome = newGenome;
        } else {
            /* Otherwise we just copy and paste mostFitParent -> child */
            child.genome = mostFitParent.genome;
        }
        /* Roll for mutation */
        if ((rand() % MUTATION_RATE_OUTOF) <= MUTATION_RATE_PERCENT){
            child.genome = mutate(child.genome);
        }

        populationToAddChildren.push_back(child);
    }
}

vector<Gene> createNewGeneration(vector<Gene> oldPopulation, int numChildren, int numMates){
    vector<Gene> newGeneration;

    for (int m = 0; m < numMates; m++){
        TwoGenes chosenCouple = chooseMates(oldPopulation);
        if (RAT_COMBINATION_METHOD =="keith"){
        	keithsReproduce(chosenCouple.first, chosenCouple.second, numChildren, newGeneration);
        }
        else {
        	newGeneration = reproduce(chosenCouple.first, chosenCouple.second, numChildren);
        }
    }

    return newGeneration;
}

double findAverageFitness(vector<Gene> & pool)
{
    int sum = 0;
    double avg;
    for (int i = 0; i < pool.size(); i++)
        sum += pool[i].fitness;

    avg = sum/pool.size();
    return avg;

}
void keithsReproduceForGenerations(string mapS, int startRow, int startCol, int numGenerations)
{
    int maxfitness;
    string maxgenome;
    /* Our first N randomized genes to get the ball rolling, run it and choose mates */
    vector<Gene> firstNGenes = makeGeneVector(INITIAL_POPULATION);
    if (USE_SEEDS){
    	vector<Gene> seeds = seedGenes();
    	firstNGenes.insert(firstNGenes.end(), seeds.begin(), seeds.end());
    }
    vector<Gene> children;
    runPopThroughMaze(mapS, startRow, startCol, firstNGenes);
    cout << "Generation " << 1 << " has population of " << firstNGenes.size() << endl;

    TwoGenes initialMates = chooseMates(firstNGenes);

    /* Roll the ball */
    TwoGenes mates = initialMates;
    children = firstNGenes;


    for(int k = 2; k <= numGenerations; k++) {
        children = createNewGeneration(children, NUM_CHILDREN, NUM_COUPLES_TO_PICK);
        runPopThroughMaze(mapS, startRow, startCol, children);
        for (int i = 0; i < children.size(); i++)
        {
            if (children[i].fitness > maxfitness) {
                maxfitness = children[i].fitness;
                maxgenome = children[i].genome;
            }
        }
        cout << "Generation " << k << " has population of " << children.size() << " with avg fitness of " << findAverageFitness(children) << " and max fitness of " << maxfitness << endl;
    }
    cout << "-------------------------------------------" << endl;
    cout << "Keith's version" << endl;
    cout << "Best rat has fitness of " << maxfitness << " after " << numGenerations << " generations" << endl;
    cout << "Its genome is: " << maxgenome << endl;
    cout << "-------------------------------------------" << endl;
}
bool ratSorter(Gene first, Gene second){ return first.fitness > second.fitness; }

vector<Gene> prunePopulation(vector<Gene> population, int percent, int cutoff){
	vector<Gene> newPopulation;
	int numberToSave = population.size() * cutoff/100;
	int numberToPrune = (population.size() - numberToSave) * percent/100;
	//array<int,population.size()> shuffleArray;
	//cout << "creating shuffle Array" << endl;
	sort(&population[0],&population[population.size()], ratSorter);
	int shuffleArray[population.size()];
	for(int i = 0; i < population.size() - numberToSave; i++){
		shuffleArray[i] = i + numberToSave;
	}
	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//cout << "shuffling..." << endl;
	random_shuffle(&shuffleArray[0],&shuffleArray[population.size()]);
	//int newPopulationPosition = 0;
	//cout << "choosing survivors..." << endl;
	for(int i = 0; i < population.size(); i++){
		//cout << "testing rat..." << endl;
		if(i <= numberToSave){
			//cout << "top rat" << endl;
			newPopulation.push_back(population[i]);
			//newPopulationPosition++;
		}
		else{
			bool prune = false;
			for(int j = 0; j < numberToPrune; j++){
				if (i == shuffleArray[j]){
					//cout << "unfit rat" << endl;
					prune = true;
				}
			}
			if(!prune){
				//cout << "fit rat" << endl;
				newPopulation.push_back(population[i]);
				//newPopulation[newPopulationPosition] = population[i];
				//newPopulationPosition++;
			}
		}
	}
	return newPopulation;
}

void trevorsReproduceForGenerations(string mapS, int startRow, int startCol, int numGenerations)
{
    int maxfitness;
    string maxgenome;
    /* Our first N randomized genes to get the ball rolling, run it and choose mates */
    vector<Gene> population = makeGeneVector(INITIAL_POPULATION);
    if (USE_SEEDS){
    	vector<Gene> seeds = seedGenes();
    	population.insert(population.end(), seeds.begin(), seeds.end());
    }
    vector<Gene> children;
    vector<Gene> random;
    runPopThroughMaze(mapS, startRow, startCol, population);
    cout << "Generation " << 1 << " has population of " << population.size() << endl;

    //TwoGenes initialMates = chooseMates(population);

    /* Roll the ball */
    //TwoGenes mates = initialMates;
    //population = population;


    for(int k = 2; k <= numGenerations; k++) {
    	//cout << "pruning..." << endl;
    	population = prunePopulation(population, PRUNE_PERCENT, PRUNE_CUTOFF);
    	//cout << "breeding..." << endl;
        children = createNewGeneration(population, NUM_CHILDREN, NUM_COUPLES_TO_PICK);
        //cout << "adding random rats..." << endl;
        random = makeGeneVector(NUM_RANDOM_RATS);
        children.insert(children.end(), random.begin(), random.end());
        //cout << "evaluating..." << endl;
        runPopThroughMaze(mapS, startRow, startCol, children);
        //cout << "integrating..." << endl;
        population.insert(population.end(), children.begin(), children.end());

        for (int i = 0; i < population.size(); i++)
        {
            if (population[i].fitness > maxfitness) {
                maxfitness = population[i].fitness;
                maxgenome = population[i].genome;
            }
        }
        cout << "Generation " << k << " has population of " << population.size() << " with avg fitness of " << findAverageFitness(population) << " and max fitness of " << maxfitness << endl;
    }
    cout << "-------------------------------------------" << endl;
    cout << "Trevor's version" << endl;
    cout << "Best rat has fitness of " << maxfitness << " after " << numGenerations << " generations" << endl;
    cout << "Its genome is: " << maxgenome << endl;
    cout << "-------------------------------------------" << endl;
}


/* -----------------------------------------------------------------------
 * reproduceForGenerations(int n):
 *
 * Given a male and female gene, reproduce numChildren children for n generations. Then
 * return a vector of Genes of the children. Will introduce random rare
 * mutations.
 * -----------------------------------------------------------------------
 */
void reproduceForGenerations(string mapS, int startRow, int startCol, int numGenerations)
{
    int maxfitness;
    string maxgenome;
    /* Our first N randomized genes to get the ball rolling, run it and choose mates */
    vector<Gene> firstNGenes = makeGeneVector(INITIAL_POPULATION);
    vector<Gene> children;
    runPopThroughMaze(mapS, startRow, startCol, firstNGenes);
    TwoGenes initialMates = chooseMates(firstNGenes);

    /* Roll the ball */
    TwoGenes mates = initialMates;

    for(int k = 0; k < numGenerations; k++) {
        children = reproduce(mates.first, mates.second, 200);
        runPopThroughMaze(mapS, startRow, startCol, children);
        mates = chooseMates(children);                            /* Incest... just sayin' */
    }

    for (int i = 0; i < children.size(); i++)
    {
        if (children[i].fitness > maxfitness) {
            maxfitness = children[i].fitness;
            maxgenome = children[i].genome;
            }

        cout << "Rat fitness: " << children[i].fitness << " and genome = " << children[i].genome << endl;
    }
    cout << "-------------------------------------------" << endl;
    cout << "Best rat has fitness of " << maxfitness << " after " << numGenerations << " generations" << endl;
    cout << "Its genome is: " << maxgenome << endl;
    cout << "-------------------------------------------" << endl;
}

/* KEITH'S NOTE I mistakenly use GENES instead of CHROMOSOMES. Biology 101 hahahaha. Pretend that genes are long strings of ASCII */
/* ANDY'S NOTE I mistakenly use x for row and y for column throughout the code */
int main(void){
    /* Generate rand seed. ONLY CALL ONCE IN THE PROGRAM! */
    srand(time(NULL));
    //string mapseed = "25:25:..$.$.X.............X....$X.X*..X$..X...*X$..$...X$.$......X.$.X...XX.$.X*.*.*..X..X.**.......X..$$$...........XX.....................$...X...*.$..X..$X..........$.*..X.....$.X..$*.$X......$...X.*X$......$.**.X.X..XX$X..*....*..X.X....$...X...X........$.X....$...*...X$*........X..$*$$......$$...$*..X.$.$......$.$.$...$..X.*.....X..$......$.XX*..X.$.X......X$*.**.....X*...$..XX..X.....$....X....X...X....X.$X$..X..........$...*.X$..X...$*...........*....XXX$$.$.$..*$XX..XX..*.....$......X.XX$..$$..X$.XX.$$..X.*..*......X......$..$.$$..*...X.........$X....$X.$$.*.$.$.$..**.....X.$.$X.*.$.........$**..X.X.X$X.$.*X.X*..$*.";
    //string mapseed = "25:25:..$.$.X........X....X....$X.X*..X$..X...*X$..X...X$.$......X.$.X...XX.$.X*.*.*..X..X.X*.......X..$$$...........XX.....$X$$$X$$$$......$...X...*.$..X..$X...X......$.*..X.....$.X..$*.$X.$$$..$...X.*X$..$...$.**.X.X..XX$X..*....*..X.X....$...X...X........$.X....$...*...X$*........X..$*$$......XX...$*..X.$.$......$.$.$...$..X.*.....X..$......$.XX*..X.$.X......X$*.**.....X*...$..XX..X.....$....X....X...X....X.$X$..X..........$...*.X$..X...$*...........*....XXX$$.$.$..*$XX..XX..*.....$......X.XX$..$$..X$.XX.$$..X.*..*......X......$..$.$$..*...X.........$X....$X.$$.*.$.$.$..**.....X.$.$X.*.$.........$**..X.X.X$X.$.*X.X*..$*.";
    //string mapseed = "25:25:..$.$.$.............$....$$.$$..$$..$...$$$..$...$$.$......$.$.$...$$.$.$$.$.$..$..$.$$.......$..$$$...........$$.....................$...$...$.$..$..$$..........$.$..$.....$.$..$$.$$......$...$.$$$......$.$$.$.$..$$$$..$....$..$.$....$...$...$........$.$....$...$...$$$........$..$$$$......$$...$$..$.$.$......$.$.$...$..$.$.....$..$......$.$$$..$.$.$......$$$.$$.....$$...$..$$..$.....$....$....$...$....$.$$$..$..........$...$.$$..$...$$...........$....$$$$$.$.$..$$$$..$$..$.....$......$.$$$..$$..$$.$$.$$..$.$..$......$......$..$.$$..$...$.........$$....$$.$$.$.$.$.$..$$.....$.$.$$.$.$.........$$$..$.$.$$$.$.$$.$$..$$.";
    string mapseed = "25:25:$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$";
    int start_row = 12;
    int start_col = 12;

    if (RAT_SELECTION_METHOD == "keith"){
    	keithsReproduceForGenerations(mapseed, start_row, start_col, NUM_GENERATIONS);
    }
    else if (RAT_SELECTION_METHOD == "trevor"){
    	trevorsReproduceForGenerations(mapseed, start_row, start_col, NUM_GENERATIONS);
    }
    else{
    	reproduceForGenerations(mapseed, start_row, start_col, NUM_GENERATIONS);
    }
    return 0;
}

/*best rat so far: 110

py9ace2K6nP:b]N4Bc2B`b_,g462*NcK\0y<n^2Qofek4O2ja63;A?t^2@fDRD>9;i`yOj0j3tV89p*lmcarXBT:UKWH0>7UQMqrRtIChT5*igzW@=M:QIq+GDJ+MV<FKMTOh7D0JNoofbAKEY3N\PcLZ`>/VI8b]l4g=ip5rH?OKGY0UKUWg+R2=j\h7Hm3<PE3aMf?ZU1/FTw.k-[I9P/BY[Z8v2+-TXrU=S@4oTP8\m5I<lLs?c*jo\GYQ4rlxNd2]w*?c5z:`2^2>iB_4tPz?R0Yz,og_V
*/
