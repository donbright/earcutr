'use strict';  /*eslint comma-spacing: 0, no-unused-vars: 0 */ /*global earcut:false */

function mkfunc(name) {
    return function() {
		loadTestData( name );
	}
};

	var minX = Infinity,
   	 maxX = -Infinity,
   	 minY = Infinity,
   	 maxY = -Infinity;
var ratio=1;

function prepCanv( testPoints ) {

	for (var i = 0; i < testPoints[0].length; i++) {
   	 minX = Math.min(minX, testPoints[0][i][0]);
   	 maxX = Math.max(maxX, testPoints[0][i][0]);
   	 minY = Math.min(minY, testPoints[0][i][1]);
   	 maxY = Math.max(maxY, testPoints[0][i][1]);
	}

	var width = maxX - minX,
   	 height = maxY - minY;

	canvas.width = window.innerWidth * 0.9;
	canvas.height = canvas.width * height / width + 10;

	ratio = (canvas.width - 10) / width;

	if (devicePixelRatio > 1) {
   	 canvas.style.width = canvas.width + 'px';
   	 canvas.style.height = canvas.height + 'px';
   	 canvas.width *= 2;
   	 canvas.height *= 2;
   	 ctx.scale(2, 2);
	}

	ctx.lineJoin = 'round';
};


function loadTestData( name ) {
	testPoints = TestData[name];
	prepCanv( testPoints );
	triangles = [];
    ctx.clearRect(0,0,canvas.width,canvas.height);

	var data = earcut.flatten(testPoints);

	console.time('earcut');
	var result = earcut(data.vertices, data.holes, data.dimensions);
	console.timeEnd('earcut');
	console.log('result',result);

	for (var i = 0; i < result.length-1; i++) {
	    var index = result[i];
		console.log(i,result[i]);
	    triangles.push([data.vertices[index * data.dimensions], data.vertices[index * data.dimensions + 1]]);
	}
	console.log('tris',triangles);
	animhelpDrawframe(60);
}

function animhelpDrawframe(ticks)
{
	var numtodraw = triangles.length * ( ticks/60 );
	for (var i = 0; triangles && i < numtodraw; i += 3) {
		//var fs = 'yellow';
		var fs = 'hsla('+360*(i/triangles.length)+',100%,50%,0.05)';
		console.log('fs',i,triangles.length,fs);
	    drawPoly([triangles.slice(i, i + 3)], 'black', fs, 1);
	};
	drawPoly(testPoints, 'black', null, 2);
}

function drawPoint(p, color, s) {
    var x = (p[0] - minX) * ratio + 5,
        y = (p[1] - minY) * ratio + 5;
    ctx.fillStyle = color || 'grey';
    ctx.fillRect(x - 3, y - 3, s, s);
}

function drawPoly(rings, color, fill, w) {

    ctx.strokeStyle = color;
    if (fill) ctx.fillStyle = fill;
	ctx.lineWidth = w;

    for (var k = 0; k < rings.length; k++) {
	    ctx.beginPath();
        var points = rings[k];
        for (var i = 0; i < points.length; i++) {
            var x = (points[i][0] - minX) * ratio + 5;
            var y = (points[i][1] - minY) * ratio + 5;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
	    ctx.stroke();
    }

    if (fill) ctx.fill('evenodd');
}


// TestData, from testoutput.js
for (var filename in TestData) {
    var menu = document.getElementById("menu");
    var textnode = document.createElement("a");
    textnode.innerHTML = filename;
    textnode.addEventListener('click', mkfunc(filename) );
    menu.appendChild(textnode);
};

var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var triangles = [];
var testPoints = [];
var testPoints  = [[[661,102],[661,96],[666,96],[666,87],[743,87],[771,87],[771,114],[750,114],[750,113],[742,113],[742,106],[710,106],[710,113],[666,113],[666,112]]];
loadTestData( 'building.json' );

