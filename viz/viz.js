'use strict';  /*eslint comma-spacing: 0, no-unused-vars: 0 */ /*global earcut:false */


var testPoints  = [[[661,102],[661,96],[666,96],[666,87],[743,87],[771,87],[771,114],[750,114],[750,113],[742,113],[742,106],[710,106],[710,113],[666,113],[666,112]]];

testPoints = testFiles["outside-ring.json"];

var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');

var minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;

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

var ratio = (canvas.width - 10) / width;

if (devicePixelRatio > 1) {
    canvas.style.width = canvas.width + 'px';
    canvas.style.height = canvas.height + 'px';
    canvas.width *= 2;
    canvas.height *= 2;
    ctx.scale(2, 2);
}

var data = earcut.flatten(testPoints);

console.time('earcut');
var result = earcut(data.vertices, data.holes, data.dimensions);
console.timeEnd('earcut');

var triangles = [];
for (i = 0; i < result.length; i++) {
    var index = result[i];
    triangles.push([data.vertices[index * data.dimensions], data.vertices[index * data.dimensions + 1]]);
}

ctx.lineJoin = 'round';

function animhelpDrawframe(ticks)
{
	var numtodraw = triangles.length * ( ticks/60 );
	for (i = 0; triangles && i < numtodraw; i += 3) {
		console.log(i,triangles[i],triangles[i+1],triangles[i+2]);
		//var fs = 'yellow';
		var fs = 'hsla('+360*(i/triangles.length)+',100%,50%,0.05)';
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
            var x = (points[i][0] - minX) * ratio + 5,
                y = (points[i][1] - minY) * ratio + 5;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
	    ctx.stroke();
    }

    if (fill) ctx.fill('evenodd');
}


