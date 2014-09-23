#include "ofApp.h"

ofMesh makeGrid(ofRectangle square, int nHoriz, int nVert) {
	ofMesh mesh;
	mesh.setMode(OF_PRIMITIVE_TRIANGLES);
	for (int j = 0; j < nVert; j++){
    for (int i = 0; i < nHoriz; i++){
			float x = ofMap(i, 0, nHoriz-1, square.x, square.x + square.width);
			float y = ofMap(j, 0, nVert-1, square.y, square.y + square.height);
			mesh.addVertex(ofPoint(x,y));
		}
	}
	for ( unsigned int y = 0; y < (nVert-1); y++ ) {
		for ( unsigned int x = 0; x < nHoriz-1; x++ ) {
			unsigned int nRow1 = y * nHoriz;
			unsigned int nRow2 = (y+1) * nHoriz;
			mesh.addIndex(nRow1 + x);
			mesh.addIndex(nRow2 + x + 1);
			mesh.addIndex(nRow1 + x + 1);
			mesh.addIndex(nRow1 + x);
			mesh.addIndex(nRow2 + x);
			mesh.addIndex(nRow2 + x + 1);
		}
	}
	return mesh;
}


ofMesh makeAnimatedGrid (ofRectangle square, int nHoriz, int nVert) {
	ofMesh mesh;
	mesh.setMode(OF_PRIMITIVE_TRIANGLES);
    float xexponent = 0.75 + 0.25*sin(ofGetElapsedTimeMillis()/1000.0);
    float yexponent = 0.75 + 0.25*cos(ofGetElapsedTimeMillis()/1000.0);
    
	for (int j = 0; j < nVert; j++){
        for (int i = 0; i < nHoriz; i++){
            float fx = ofMap(i, 0, nHoriz-1, 0,1);
            float fy = ofMap(j, 0, nVert-1,  0,1);
            
            fx = powf(fx, xexponent);
            fy = powf(fy, yexponent);
            
			float x = ofMap(fx, 0,1, square.x, square.x + square.width);
			float y = ofMap(fy, 0,1, square.y, square.y + square.height);
			mesh.addVertex(ofPoint(x,y));
		}
	}
    
	for ( unsigned int y = 0; y < (nVert-1); y++ ) {
		for ( unsigned int x = 0; x < nHoriz-1; x++ ) {
			unsigned int nRow1 = y * nHoriz;
			unsigned int nRow2 = (y+1) * nHoriz;
			mesh.addIndex(nRow1 + x    );
			mesh.addIndex(nRow2 + x + 1);
			mesh.addIndex(nRow1 + x + 1);
			mesh.addIndex(nRow1 + x    );
			mesh.addIndex(nRow2 + x    );
			mesh.addIndex(nRow2 + x + 1);
		}
	}
	return mesh;
}



void ofApp::setup(){
    
    gridRes = 13;
    
	ofSetVerticalSync(true);
	ofMesh mesh = makeAnimatedGrid (ofRectangle(100,100,600,600), gridRes,gridRes);
	puppet.setup(mesh);
	
	puppet.setControlPoint(0); // pin the top left
	puppet.setControlPoint(9); // pin the top right
}



void ofApp::update(){
    
    bool bUseAnimated = true;
    if (bUseAnimated){
        
        ofMesh mesh = makeAnimatedGrid (ofRectangle(100,100,600,600), gridRes,gridRes);
        puppet.setup(mesh);
        puppet.setControlPoint(0); // pin the top left
        puppet.setControlPoint(9); // pin the top right
    }
    
    
	puppet.update();
    

    
}

void ofApp::draw(){
	ofBackground(0);
	puppet.drawWireframe();
	puppet.drawControlPoints();
}





void ofApp::testAccelerate(){
    
    //===============================
    // Test of matrix inversion, from:
    // http://forums.macnn.com/79/developer-center/364946/using-the-accelerate-framework/
    // Sample problem to solve Ax=b for x. System of equations will be:
    // A=[[1. 2. 3. 4.]
    //    [5. 6. 7. 8.]
    //    [9. 10. 11. 12.]
    //    [13. 14. 15. 16.]]
    //
    // b = [17. 18. 19. 20.]
    int num_rows = 4;
    int num_cols = 4;
    int num_elts = num_rows*num_cols;
    float * A;
    A = new float[num_elts];
    for (int i=0; i<num_elts; i++){
        A[i]=0.0;
    }
    
    // Fill in A the "normal" way
    float val=1.0;
    for (int row=0; row<num_rows; row++) {
        for (int col=0; col<num_cols; col++) {
            A[row+col*num_rows]=val;
            val+=1.0;
        }
    }
    
    // display A:
    cout << "A=" << endl;
    for (int row=0; row<num_rows; row++) {
        for (int col=0; col<num_cols; col++) {
            cout << A[row+col*num_rows] << " ";
        }
        cout << endl;
    }
    cout << endl;
    
    // Create data array for b. Will be a vector that is num_rows in size.
    float *b;
    b = new float[num_rows];
    
    // zero out B:
    for (int i=0; i<num_rows; i++){
        b[i]=0.0;
    }
    
    // fill in B:
    for (int i=0; i<num_rows; i++) {
        b[i]=val;
        val+=1.0;
    }
    
    // display b.
    cout << "b=" << endl;
    for(int i=0; i<num_rows; i++) {
        cout << b[i] << endl;
    }
    cout << endl;
    
    // create x vector with which will be our solution vector
    float *x = new float[num_rows];
    for (int i=0; i<num_rows; i++) {
        x[i]=b[i];	// copy b into x, as the lapack routines do an inplace solve
    }	// i.e.- the rhs vector input, is overwritten with the solution vector
    // so if we want to preserve b, we need to make a copy.
    
    //-------------------------//
    // The Solver- using dgesv //
    //-------------------------//
    __CLPK_integer n, lda, ldb, nrhs, info;	// Need some parameters/storage vars of type clapack integer
    n=lda = ldb = num_rows;	// set n, lda, ldb to num_rows
    nrhs=1;	// we are solving for 1 rhs vector
    __CLPK_integer * ipiv = new __CLPK_integer[num_rows];	// create a pivot vector for the solver
    sgesv_ (&n, &nrhs, A, &lda, ipiv, x, &ldb, &info);	// perform the [d]ouble precision [ge]eneral Matrix [s]ol[v]e (dgesv)
    
    // output x
    cout << "x=" << endl;
    for (int i=0; i<num_rows; i++) {
        cout << x[i] << endl;
    }
    
    
}









