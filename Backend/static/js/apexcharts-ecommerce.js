/*
-------------------------------------------------------------------------
* Template Name    : Snow - Tailwind CSS Admin & Dashboard Template     * 
* Author           : Webonzer                                           *
* Version          : 1.0.0                                              *
* Created          : March 2023                                         *
* File Description : Main JS file of the template                       *
*------------------------------------------------------------------------
*/
var revenue = {
    chart: {
        height: 250,
        type: "area",
        fontFamily: "Inter, sans-serif",
        zoom: {
            enabled: false,
        },
        toolbar: {
            show: false,
        },
    },
    series: [
        {
            name: "Current Week",
            data: [0, 1000, 5000, 10000, 8000, 11000, 15000],
        },
        {
            name: "Previous Week",
            data: [2000, 3000, 6000, 12000, 9000, 13000, 14000],
        },
    ],
    dataLabels: {
        enabled: false,
    },
    stroke: {
        show: true,
        curve: "smooth",
        width: 3,
        lineCap: "square",
    },
    dropShadow: {
        enabled: true,
        opacity: 0.2,
        blur: 10,
        left: -7,
        top: 22,
    },
    colors: ["#1C1C1C", "#A8C5DA"],
    markers: {
        discrete: [
            {
                seriesIndex: 0,
                dataPointIndex: 4,
                fillColor: "#1C1C1C",
                strokeColor: "#fff",
                size: 6,
            },
            {
                seriesIndex: 1,
                dataPointIndex: 5,
                fillColor: "#A8C5DA",
                strokeColor: "#fff",
                size: 6,
            },
        ],
    },
    labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    xaxis: {
        axisBorder: {
            show: false,
        },
        axisTicks: {
            show: false,
        },
        crosshairs: {
            show: true,
        },
        labels: {
            offsetX: 0,
            offsetY: 5,
            style: {
                fontSize: "12px",
                cssClass: "apexcharts-xaxis-title",
            },
        },
    },
    yaxis: {
        tickAmount: 5,
        labels: {
            formatter: (value) => {
                return value / 1000 + "M";
            },
            offsetX: -10,
            offsetY: 0,
            style: {
                fontSize: "12px",
                cssClass: "apexcharts-yaxis-title",
            },
        },
        opposite: false,
    },
    grid: {
        borderColor: "#e0e6ed",
        strokeDashArray: 7,
        xaxis: {
            lines: {
                show: false,
            },
        },
        yaxis: {
            lines: {
                show: true,
            },
        },
        padding: {
            top: 0,
            right: 0,
            bottom: 0,
            left: 0,
        },
    },
    legend: {
        show: false,
    },
    tooltip: {
        marker: {
            show: true,
        },
        x: {
            show: false,
        },
    },
    fill: {
        type: "gradient",
        gradient: {
            shadeIntensity: 1,
            inverseColors: !1,
            opacityFrom: 0,
            opacityTo: 0,
            stops: [100, 100],
        },
    },
};
var chart6 = new ApexCharts(document.querySelector("#revenue"), revenue);
chart6.render();

//Task Ovrview Chart
var taskovrview = {
    series: [
        {
            name: "Task 1",
            data: [12, 18, 15, 25, 6, 18],
        },
        {
            name: "Task 2",
            data: [4, 2, 4, 4, 2, 4],
        },
    ],
    chart: {
        height: 156,
        type: "bar",
        toolbar: {
            show: false,
        },
        stacked: true,
    },
    dataLabels: {
        enabled: false,
    },
    stroke: {
        show: true,
        width: 1,
    },
    colors: ["#93BFDF", "#CFDFEB"],
    yaxis: {
        tickAmount: 4,
        opposite: false,
    },
    xaxis: {
        categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    },

    fill: {
        opacity: 1,
    },
    plotOptions: {
        bar: {
            horizontal: false,
            columnWidth: "25%",
        },
    },
    legend: {
        show: false,
    },
};
var chart7 = new ApexCharts(
    document.querySelector("#taskovrview"),
    taskovrview
);
chart7.render();

//Total Sales chart bar
var totalsales = {
    series: [300.56, 135.18, 48.96, 154.02],
    chart: {
        type: "donut",
        height: 300,
        fontFamily: "Inter, sans-serif",
    },
    dataLabels: {
        enabled: false,
    },
    legend: {
        position: "bottom",
        horizontalAlign: "left",
        fontSize: "12px",
        formatter: function (val, opts) {
            return val + " - " + "$" + opts.w.globals.series[opts.seriesIndex];
        },
        markers: {
            width: 6,
            height: 6,
            offsetX: -5,
        },
        height: 100,
        width: 160,
        offsetY: 0,
    },

    colors: ["#BAEDBD", "#C6C7F8", "#1C1C1C", "#95A4FC"],
    labels: ["Direct", "Affilliate", "Sponsored", "E-mail"],
    states: {
        hover: {
            filter: {
                type: "none",
                value: 0.15,
            },
        },
        active: {
            filter: {
                type: "none",
                value: 0.15,
            },
        },
    },
};
var chart8 = new ApexCharts(document.querySelector("#totalsales"), totalsales);
chart8.render();
