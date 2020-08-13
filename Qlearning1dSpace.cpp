#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <iomanip>

#ifdef __linux__
#include <unistd.h>
#define sleep(x) usleep(x * 1000);
#elif _WIN32
#include <Windows.h>
#define sleep(x) Sleep(x)
#else
#define sleep(x) cout << '';
#endif


using namespace std;

/*--------------------
9x1 map

all state 27

g=gold, p=player, t=trap
--------------------*/

typedef struct statesGP {
	int g, p;
}STATES_GP;

typedef struct statesGPT {
	int g, p, t;
}STATES_GPT;

class Agent;
class Space1d;


class Space1d
{
public:
	void printSpace();
	int checkCollision();
	void setRandomMap();
	void setMap(STATES_GPT inputState);
	STATES_GP getStateGP();
	STATES_GPT getStateGPT();
	void moveLeft();
	void moveRight();

private:
	int g, p, t;
};
void Space1d::printSpace()
{
	int i;
	for (i = 0; i < 9; i++) {
		if (i == p) cout << "  P  ";
		else if (i == g) cout << " GOLD ";
		else if (i == t) cout << " trap ";
		else cout << " . ";
	}
	cout << endl;
}
void Space1d::setRandomMap() {
	if (rand() % 2) {
		//left gold
		g = rand() % 3;
		//player
		p = rand() % 3 + 3;
		//right trap
		t = rand() % 3 + 6;
	}
	else {
		//left trap
		t = rand() % 3;
		//player
		p = rand() % 3 + 3;
		//right gold
		g = rand() % 3 + 6;
	}
}
void Space1d::setMap(STATES_GPT inputState)
{
	g = inputState.g;
	p = inputState.p;
	t = inputState.t;
}
int Space1d::checkCollision() {
	//-1 trapped
	if (p == t) return -1;
	//1 get gold
	else if (p == g) return 1;
	//wrong
	else if (p == t) return -2;
	//0 normal
	return 0;
}
STATES_GP Space1d::getStateGP()
{
	STATES_GP state;
	state.g = g;
	state.p = p;
	return state;
}
STATES_GPT Space1d::getStateGPT()
{
	STATES_GPT state;
	state.g = g;
	state.p = p;
	state.t = t;
	return state;
}
void Space1d::moveLeft()
{
	p--;
}
void Space1d::moveRight()
{
	p++;
}

class Agent
{
public:
	// common train
	Agent(Space1d *space1d, double gammaInput, double alphaInput);
	void addState(STATES_GPT state);
	void addAction(int move);
	void addReward(double reward);
	void train(bool isPrintSpace, bool isUpdatable, bool isDelayed);
	bool getIsSucceed();
	double getTotalReward();
	int getStepCount();
	double getMoveRatio();
	double getEpsilon();
    void resetQ();
	// only debug
	void printQ();

private:
	Space1d *mySpace1d;
	double reward, moveRatio, epsilon;
	int step;
	vector < STATES_GPT > states;
	vector < int > actions;
	vector < double > rewards;
	//discout factor
	double gamma;
	//learning rate
	double alpha;
	// [statePlayer][goldLocation][trapLocation][action] = Q probability
	double Qtable[9][9][9][2];
	double maxQ(int state);
	bool isSucceed;
	void updateQ();
	void saveReplay();
	void resetReplay();
	void randomMove();
	void modelMove();
	void moveAgent();
};
Agent::Agent(Space1d *space1d ,double gammaInput, double alphaInput)
{

    mySpace1d = space1d;
	reward = 0;
	moveRatio = 0;
	epsilon = 1.0;
	step = 0;
	gamma = gammaInput;
	alpha = alphaInput;
	isSucceed = false;

	resetReplay();
    resetQ();
}
void Agent::resetQ()
{
    int i, j, k, l;
    for (i = 0; i < 9; i++)
		for (j = 0; j < 9; j++)
			for (k = 0; k < 9; k++)
				for (l = 0; l < 2; l++)
                    Qtable[i][j][k][l] = 0.0;

}
void Agent::addState(STATES_GPT state)
{
	states.push_back(state);
}
void Agent::addAction(int move)
{
	actions.push_back(move);
}
void Agent::addReward(double reward)
{
	rewards.push_back(reward);
}
double Agent::maxQ(int state)
{
	//cout << Qtable[state][mySpace1d->getStateGPT().g][mySpace1d->getStateGPT().t][0] << " " << Qtable[state][mySpace1d->getStateGPT().g][mySpace1d->getStateGPT().t][1] <<endl;

	if (Qtable[state][mySpace1d->getStateGPT().g][mySpace1d->getStateGPT().t][0] > Qtable[state][mySpace1d->getStateGPT().g][mySpace1d->getStateGPT().t][1]) {
		return Qtable[state][mySpace1d->getStateGPT().g][mySpace1d->getStateGPT().t][0];
	}
	else if (Qtable[state][mySpace1d->getStateGPT().g][mySpace1d->getStateGPT().t][0] < Qtable[state][mySpace1d->getStateGPT().g][mySpace1d->getStateGPT().t][1]) {
		return Qtable[state][mySpace1d->getStateGPT().g][mySpace1d->getStateGPT().t][1];
	}
	else
	{
		//cout << "new update" <<endl;
		return Qtable[state][mySpace1d->getStateGPT().g][mySpace1d->getStateGPT().t][0];
	}
	// error
	return 0;
}
void Agent::updateQ() {
	int i;
	int stateP, action;
	double temp1, temp2, reward;
	for (i = 0; i < (int)actions.size(); i++) {
		/*
		i =  0,1,2,3
						 Start End
						   S   E
		replay Rewards   0 1 2 3
		replay  States     0 1 2 3
		replay actions     0 1 2

		*/
		reward = rewards[i + 1];
		stateP = states[i].p;
		action = actions[i];

		//g, p = fixed
		temp1 = (1.0 - alpha) * Qtable[stateP][states[0].g][states[0].t][action];
		temp2 = (alpha * (reward + gamma * maxQ(states[i + 1].p)));
		Qtable[stateP][states[0].g][states[0].t][action] = temp1 + temp2;

	}
}
void Agent::printQ() {
	int i;
	for (i = 0; i < 9; i++) {
		cout << setw(5) << Qtable[i][states[0].g][states[0].t][0] << " ";
		cout << setw(5) << Qtable[i][states[0].g][states[0].t][1] << " ";
		if (Qtable[i][states[0].g][states[0].t][0] > Qtable[i][states[0].g][states[0].t][1])
		{
			cout << setw(5) << " left ";
		}
		else if (Qtable[i][states[0].g][states[0].t][0] < Qtable[i][states[0].g][states[0].t][1]) {
			cout << setw(5) << " right ";
		}
		else {
			cout << setw(5) << " random ";
		}
		cout << endl;
	}
	cout << endl;
}
void Agent::saveReplay() {
	STATES_GPT stateGPTtmp = mySpace1d->getStateGPT();
	states.push_back(stateGPTtmp);
	rewards.push_back(reward);
}
void Agent::resetReplay()
{
	rewards.clear();
	states.clear();
	actions.clear();
}
void Agent::randomMove() {
	int move;
	move = rand() % 2;
	switch (move) {
	case 0:
		mySpace1d->moveLeft();
		if (mySpace1d->getStateGP().p > 9 || mySpace1d->getStateGP().p < 0)
		{
			mySpace1d->moveRight();
			mySpace1d->moveRight();
			move = 1;
		}
		break;
	case 1:
		mySpace1d->moveRight();
		if (mySpace1d->getStateGP().p > 9 || mySpace1d->getStateGP().p < 0)
		{
			mySpace1d->moveLeft();
			mySpace1d->moveLeft();
			move = 0;
		}
		break;
	}
	//save action
	addAction(move);
}
void Agent::modelMove() {
	int move = -1;
	if (Qtable[states.back().p][states[0].g][states[0].t][0] == Qtable[states.back().p][states[0].g][states[0].t][1]) {
		randomMove();
		return;
	}
	else if (Qtable[states.back().p][states[0].g][states[0].t][0] > Qtable[states.back().p][states[0].g][states[0].t][1]) {
		move = 0;
	}
	else if (Qtable[states.back().p][states[0].g][states[0].t][0] < Qtable[states.back().p][states[0].g][states[0].t][1]) {
		move = 1;
	}
	switch (move) {
	case 0:
		mySpace1d->moveLeft();

		if (mySpace1d->getStateGP().p > 9 || mySpace1d->getStateGP().p < 0)
		{
			mySpace1d->moveRight();
			mySpace1d->moveRight();
			move = 1;
		}
		break;
	case 1:
		mySpace1d->moveRight();
		if (mySpace1d->getStateGP().p > 9 || mySpace1d->getStateGP().p < 0)
		{
			mySpace1d->moveLeft();
			mySpace1d->moveLeft();
			move = 0;
		}
		break;
	}
	//save action
	addAction(move);
	moveRatio += 1;
}
void Agent::moveAgent() {
	if (rand() % 10000 * 0.0001 > epsilon) {
		modelMove();
	}
	else {
		randomMove();
	}
}

void Agent::train(bool isPrintSpace, bool isUpdatable, bool isDelayed) {
	step = 0;
	moveRatio = 0;
	reward = 0;
	isSucceed = false;
	mySpace1d->setRandomMap();
	resetReplay();
	//start

	//saveStart
	saveReplay();
	while (step < 100) {
		
		reward = 0;
		if(isDelayed)sleep(144);
		//usleep(144000);
		if (isPrintSpace)mySpace1d->printSpace();
		moveAgent();
		if (mySpace1d->checkCollision() == 1) {
			//got gold
			step++;
			reward = 1.0 * pow(gamma, step);
			if (isDelayed) cout << "                                          get GOLD" << endl;
			isSucceed = true;
			break;
		}
		else if (mySpace1d->checkCollision() == -1) {
			//trap
			step++;
			reward = -1.0 * pow(gamma, step);
			if (isDelayed) cout << "                                          trapped!" << endl;
			break;
		}
		else {
			//nothing
			saveReplay();
			step++;
		}
	}
	//end loop
	moveRatio = moveRatio / (double)step;
	if (step < 100) {
		//if (isPrintSpace) mySpace1d->printSpace();
		saveReplay();
		if (isUpdatable) updateQ();
	}
	else {
		reward = 0;
		saveReplay();
		if (isUpdatable) updateQ();
	}
	if (epsilon > 0.01) epsilon -= 0.01;
}
bool Agent::getIsSucceed()
{
	return isSucceed;
}
double Agent::getTotalReward()
{
	int i;
	double totalReward = 0;
	for (i = 0; i < (int)rewards.size(); i++)
	{
		totalReward += rewards[i];
	}
	return totalReward;
}
int Agent::getStepCount()
{
	return step;
}
double Agent::getMoveRatio()
{
	return moveRatio;
}
double Agent::getEpsilon()
{
	return epsilon;
}


int main() {

    Space1d space;
	Agent agent(&space , 0.9, 0.1);


	//srand((unsigned int)time(NULL));
	int i;
	int succeedCount;
	double succeedRatio;
	int trainCount = 10000;
	int checkTerm = trainCount >= 100 ? trainCount / 100 : 0;

	//train
	i = 0;
	succeedCount = 0;
	succeedRatio = 0;
	//while( 70 > succeedRatio || i <= checkTerm) {
	while (i < trainCount) {
		agent.train(false, true, false);
		if (agent.getIsSucceed()) {
			succeedCount++;
		}
		//print every 1000times
		if (i % checkTerm == 0 || i >= trainCount)
		{
			cout << " train " << i;
			cout << " reward " << setw(11) << setprecision(3) << agent.getTotalReward();
			cout << " Modelratio " << setw(5) << setprecision(3) << agent.getMoveRatio();
			succeedRatio = (succeedCount != 0 && i + 1 != 0 ? ((double)succeedCount / (double)(i + 1))*100.0 : 0);
			cout << " accuracy " << setprecision(8) << succeedRatio << "%";
			cout << endl;
		}
		i++;

	}


	//test
	cout << endl << "------------------test--------------------" << endl;
	i = 0;
	succeedCount = 0;
	succeedRatio = 0;
	while (1) {
		agent.train(true, true, true);
		i++;
	}
	return 0;
}
