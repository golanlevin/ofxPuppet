#pragma once

#include "ofMain.h"
#include "ofxPuppetInteractive.h"

#include <Accelerate/Accelerate.h>

class ofApp : public ofBaseApp{
public:
	void setup();
	void update();
	void draw();
	
	ofxPuppetInteractive puppet;
    int gridRes;
    
    
    void testAccelerate(); 
};
