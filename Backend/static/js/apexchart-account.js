/*
-------------------------------------------------------------------------
* Template Name    : Snow - Tailwind CSS Admin & Dashboard Template     * 
* Author           : Webonzer                                           *
* Version          : 1.0.0                                              *
* Created          : March 2023                                         *
* File Description : Main JS file of the template                       *
*------------------------------------------------------------------------
*/
//Selling Categories
var sellingcategories = {
    series: [
        {
            name: "Device",
            data: [100, 300, 400, 800, 1200, 1600, 600],
        },
    ],
    chart: {
        height: 236,
        type: "bar",
        events: {
            click: function (chart, w, e) {
                // console.log(chart, w, e)
            },
        },
        toolbar: {
            show: false,
        },
    },
    colors: ["#BAEDBD", "#C6C7F8", "#1C1C1C", "#B1E3FF", "#95A4FC", "#A1E3CB", "#A8C5DA"],
    plotOptions: {
        bar: {
            columnWidth: "30%",
            distributed: true,
        },
    },
    dataLabels: {
        enabled: false,
    },
    legend: {
        show: false,
    },
    yaxis: {
        tickAmount: 6,
        labels: {
            formatter: (value) => {
                return value / 100 + "K";
            },
            offsetX: -10,

            offsetY: 0,
            style: {
                fontSize: "12px",
            },
        },
        opposite: false,
    },
    xaxis: {
        categories: ["Phones", "Laptops", "Headsets", "Games", "Keyboardsy", "Monitors", "Speakers"],
        labels: {
            style: {
                fontSize: "12px",
            },
        },
    },
};
var chart9 = new ApexCharts(document.querySelector("#sellingcategories"), sellingcategories);
chart9.render();

//Agent chart
var agentchart = {
    series: [
        {
            name: "Agents Chart",
            data: [76, 85, 101, 98, 87, 105, 91, 114, 94, 60, 50],
        },
        {
            name: "Agents",
            data: [35, 41, 36, 26, 45, 48, 52, 53, 41, 55, 40],
        },
    ],
    chart: {
        type: "bar",
        height: 222,
        toolbar: {
            show: false,
        },
    },
    plotOptions: {
        bar: {
            horizontal: false,
            columnWidth: "40%",
            endingShape: "rounded",
            borderRadius: 4,
            borderRadiusApplication: "end",
            borderRadiusWhenStacked: "last",
        },
    },
    dataLabels: {
        enabled: false,
    },
    legend: {
        show: false,
    },
    stroke: {
        show: true,
        width: 0,
        colors: ["transparent"],
    },
    colors: ["#A8C5DA", "#CFDFEB"],
    xaxis: {
        categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Aug", "Sep", "Oct", "Nov", "Dec"],
    },

    fill: {
        opacity: 1,
    },
};

var chart10 = new ApexCharts(document.querySelector("#agentchart"), agentchart);
chart10.render();

//Clients Chart
var clientschart = {
    series: [
        {
            name: "Clients Chart",
            data: [76, 85, 101, 98, 87, 105, 91, 114, 94, 60, 50],
        },
        {
            name: "Clients",
            data: [35, 41, 36, 26, 45, 48, 52, 53, 41, 55, 40],
        },
    ],
    chart: {
        type: "bar",
        height: 222,
        toolbar: {
            show: false,
        },
    },
    plotOptions: {
        bar: {
            horizontal: false,
            columnWidth: "40%",
            endingShape: "rounded",
            borderRadius: 4,
            borderRadiusApplication: "end",
            borderRadiusWhenStacked: "last",
        },
    },
    dataLabels: {
        enabled: false,
    },
    legend: {
        show: false,
    },
    stroke: {
        show: true,
        width: 0,
        colors: ["transparent"],
    },
    colors: ["#95A4FC", "#C6C7F8"],
    xaxis: {
        categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Aug", "Sep", "Oct", "Nov", "Dec"],
    },

    fill: {
        opacity: 1,
    },
};

var chart11 = new ApexCharts(document.querySelector("#clientschart"), clientschart);
chart11.render();
