var currentTestName = "";
var labelsToggle = 0;
var fillToggle = 1;
var labelmap = [];

function setupCanvas(viewerid,canvasid) {
	// size must be reset because CSS sizing doesn't work on canvases.
	canvas = document.getElementById(canvasid);
	par = document.getElementById(viewerid);
	wextra = 0;
	hextra = 0;
	var wdt = par.clientWidth - wextra;
	var ht = par.clientHeight - hextra;
	canvas.setAttribute('height',ht+"px");
	canvas.setAttribute('width',wdt+"px");

	canvas.addEventListener("wheel", function (ev) {
		console.log("wheel");
		console.log(ev.deltaX);
		console.log(ev.deltaY);
		console.log(ev.offsetX);
		console.log(ev.offsetY);
		redraw( canvas.id );
	} );

	return canvas;
}

/* Map from pset coordinates to canvas coordinates
   Example: pset has x-coordinates from -50 to 100,
   canvas x coordinates go from 0 to 600.
   consider a single point, and its x coordinate.
   first lets translate x so it goes from 0 to 150. psetx=(inputx-minx).
   now psetx / pset width is proportional to canvas x / canvas width
   ox/ow = cx/cw, therefore cx = (ox/ow)*cw
   (100-(-50))/150 * 600 => 600
   (-50-(-50))/150 * 600 => 0
   (  0-(-50))/150 * 600 => 200
   now, also cx = ox * ( cw/ow ), cw/ow can be the 'xratio', and we have
   different ratio y direction, y ratio. in order to keep the pset's
   original aspect ratio, only multiply by the lowest of these two ratios.
*/
class PointMapper {
	constructor( pset, canvas ) {
		this.canvas = canvas;
		this.bbox = findBoundingBox( pset );
		//console.log("mapper bbox",this.bbox,pset);
		this.psetw = this.bbox.maxx-this.bbox.minx;
		this.pseth = this.bbox.maxy-this.bbox.miny;
		var wratio = (this.canvas.width-1) / this.psetw;
		var hratio = (this.canvas.height-1) / this.pseth;
		this.ratio = Math.min(wratio,hratio);
	}
	x( psetx ) {
		return (psetx-this.bbox.minx)*this.ratio;
	}
	y( psety ) {
		return (psety-this.bbox.miny)*this.ratio;
	}
}

function labelmapr( mapr, x, y, offset,labeltxt ) {
	mx = mapr.x(x);
	my = mapr.y(y);
	if (!labelmap[x]) {
		labelmap[x]=[];
		labelmap[x][y] = 1;
	} else if (!labelmap[x][y]) {
		labelmap[x][y] = 1;
	} else {
		times_used = labelmap[x][y];
		my += offset * times_used;
		labelmap[x][y] += 1;
	}
	if (mx>(mapr.canvas.width-30))  { mx -= 40; }
	if (my>(mapr.canvas.height-10)) { my -= 10; }
	if (mx<(0+10)) { mx += 10; }
	if (my<(0+offset)) { my += 22; }
	return [mx,my];
}

function findBoundingBox( Pset ) {
	var minx = Pset[0][0][0], miny = Pset[0][0][1];
	var maxx = minx, maxy = miny;
	for ( var i=0; i<Pset.length; i++ ) {
		var contour = Pset[i];
		for ( var j=0; j<contour.length; j++ ) {
			var point = contour[j];
			maxx = Math.max(maxx,point[0]);
			maxy = Math.max(maxy,point[1]);
			minx = Math.min(minx,point[0]);
			miny = Math.min(miny,point[1]);
		}
	}
	return {"minx":minx,"miny":miny,"maxx":maxx,"maxy":maxy};
}

function flatten( data ) {
	vertices = [];
	dimensions = data[0][0].length;
	for (var i = 0; i < data.length; i++) {
        for (var j = 0; j < data[i].length; j++) {
            for (var d = 0; d < dimensions; d++) vertices.push(data[i][j][d]);
        }
	}
	return vertices;
}

function makeTriPset( mainPset, tris, mapr ) {
	dimensions = mainPset[0][0].length;
	vertices = flatten( mainPset );
	trianglePset = [];
	for (var i = 0; i < tris.length; i+=3) {
		triContour = [];
		for (var j = 0; j < 3; j++ ) {
		    ptindex = tris[i+j];
			x = vertices[ ptindex * dimensions + 0 ];
			y = vertices[ ptindex * dimensions + 1 ];
			triContour.push( [x,y] );
		}
		trianglePset.push( triContour );
    }
	//console.log(trianglePset);
	return trianglePset;
}

function drawPset( pset, canvas, mapr, labels, lastcontour, fillLevel, special ) {
	ctx = canvas.getContext('2d');
	labelmap = [];
	todraw = Math.min(pset.length,lastcontour);
	var pointindex = 0;
	dim = pset[0][0].length;
	for ( var i=0; i < todraw; i++,pointindex+=dim ) {
		//console.log("dpset"+i);
		var contour = pset[i];
		ctx.beginPath();
		ctx.moveTo( mapr.x(contour[0][0]), mapr.y(contour[0][1]) );
		for ( var j=0; j < contour.length; j++,pointindex+=dim ) {
			var point = contour[j];
			ctx.lineTo( mapr.x(point[0]), mapr.y(point[1]) );
			if ( labels>0 ) { 
				ctx.font="1.5em Verdana";
				ctx.fillStyle = "black";
				labeltxt = i+':'+j;
				if (labels==2) labeltxt = pointindex;
				labelpoint = labelmapr(mapr,point[0],point[1],40,labeltxt);
				ctx.fillText( labeltxt, labelpoint[0], labelpoint[1] );
				ctx.fillStyle = fs;
				//console.log(i+','+j, mapr.x(point[0]), mapr.y(point[1]));
			}
		}
		ctx.closePath();
		if (fillLevel==1) {
			//var fs = 'hsla('+360*(i*3)/(pset.length*3-1)+',100%,50%,0.05)';
			//console.log('fs',i*3,pset.length*3-1,fs );
			hue = 360*(i*3)/(pset.length*3-1)
			if (special=="earcut") { hue = (hue + 180) % 360; };
			var fs = 'hsla('+hue+',100%,50%,0.32)';
			ctx.fillStyle = fs;
			ctx.strokeStyle = "black";
			ctx.fill();
		} else if (fillLevel==2) {
			ctx.fillStyle = "gray";
			ctx.strokeStyle = ctx.fillStyle;
			ctx.fill();
		} else {
			ctx.strokeStyle = "black";
		}
		ctx.stroke();
	}

}

// create a function to respond to filename clicks, used by build_testmenu
function mkfunc( testname, canvasid ) {
    return function() {
		currentTestName = testname;
		redraw( canvasid );
    }
};

function build_testmenu( canvasid ) {
	/* this data comes from the files under viz/testoutput/*.js
	and testOutput=[] defined inside viz.html*/

	names = Object.keys(testOutput);
	names.sort();

	// fill up the 'file' menu on the left hand side of the screen
	for (var i in names) {
		testname = names[i];
	    var menu = document.getElementById("menu");
	    var filenode = document.createElement("div");
		filenode.setAttribute( "class", "file" );
		if (testOutput[testname]["pass"]=="0") {
			filenode.setAttribute( "class", "file_fail" ); }
	    filenode.innerHTML = testname;
	    filenode.addEventListener('click', mkfunc(testname,canvasid) );
	    menu.appendChild(filenode);
	}
}

function drawTest( pset, tris, report, canvas, showlabels, fillLevel, special )
{
	ctx = canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
	mapr = new PointMapper( pset, canvas );
	triPset = makeTriPset( pset, tris, mapr ); 
	drawPset( pset, canvas, mapr, showlabels, pset.length, 0 );
	if ( fillLevel == 1 ) {
			todraw = triPset.length;
			drawPset( triPset, canvas, mapr, false, todraw, fillLevel, special );
	} else if ( fillLevel == 2 ) {
			todraw = triPset.length;
			drawPset( triPset, canvas, mapr, false, todraw, fillLevel, special );
	}
	//	for ( var i = 0; i < 6; i ++ ) { 
	//	}
    var reportbox = document.getElementById("report");
	reportbox.innerHTML = report + '<br/>' + tris;
}

function redraw( canvasid ) {
	canvas = document.getElementById( "mycan" );
	rpttxt = currentTestName + '\n'+  testOutput[currentTestName]["report"];
	drawTest( testOutput[currentTestName]["json"],
			  testOutput[currentTestName]["triangles"],
			  rpttxt,
			  canvas,
			  labelsToggle,
			  fillToggle );
}
function redrawEarcutVersion ( canvasid ) {
	// this section creates earcut.js versions of each test from testOutput
	var canvas = document.getElementById( "mycan" );
	var testname = currentTestName;
	var jsondata = testOutput[testname]["json"];
	var data = earcut.flatten( jsondata );
	console.log("-------earcut start" );
	var result = earcut(data.vertices, data.holes, data.dimensions);
	console.log("-------earcut end");
	var rpttxt = "filename:" + testname + ".json\n num tris" + result.length/3;

	drawTest( jsondata,
			  result,
			  rpttxt,
			  canvas,
			  labelsToggle,
			  fillToggle,
			  "earcut" );
}


function labelsfunc( canvasid ) { labelsToggle = (labelsToggle+1)%3;redraw(canvasid);};
function trisfunc( canvasid ) { fillToggle = (fillToggle+1)%3; redraw(canvasid); };
function earcutfunc( canvasid ) { redrawEarcutVersion( canvasid ); }

function setupControls( canvasid ) {
    document.getElementById("labelsbutton").addEventListener('click',
		labelsfunc, canvasid );
    document.getElementById("trisbutton").addEventListener('click',
		trisfunc, canvasid );
    document.getElementById("earcutbutton").addEventListener('click', 
		earcutfunc, canvasid );
}

function main() {
	canvas = setupCanvas("viewer","mycan");
	setupControls( "mycan" );

	/*
	var testPset = [[[661,112],[661,96],[666,96],[666,87],[743,87],[771,87],[771,114],[750,114],[750,113],[742,113],[742,106],[710,106],[710,113],[666,113],[666,112]]];
	var testTris = [14, 0, 1, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 11, 
		12, 13, 14, 1, 2, 4, 6, 8, 11, 13, 14, 14, 2, 4, 4, 8, 10, 11, 14, 
		4, 4, 10, 11];
	var testReport = "default shape, " + testTris.length + " triangles";
	drawTest( testPset, testTris, testReport, canvas );
	*/

	build_testmenu( canvas );

	currentTestName = Object.keys(testOutput)[0];
	console.log(currentTestName);
	redraw( canvas.id );
}

main();
