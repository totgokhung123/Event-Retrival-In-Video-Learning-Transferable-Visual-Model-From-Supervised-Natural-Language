/*
-------------------------------------------------------------------------
* Template Name    : Snow - Tailwind CSS Admin & Dashboard Template     * 
* Author           : Webonzer                                           *
* Version          : 1.0.0                                              *
* Created          : March 2023                                         *
* File Description : Main JS file of the template                       *
*------------------------------------------------------------------------
*/
var projectstatus = {
    series: [67.6, 26.4, 6],
    chart: {
        type: "donut",
        height: 250,
        fontFamily: "Nunito, sans-serif",
    },
    dataLabels: {
        enabled: false,
    },
    legend: {
        position: "bottom",
        horizontalAlign: "center",
        fontSize: "12px",

        markers: {
            width: 6,
            height: 6,
            offsetX: -5,
        },
        height: 30,
        offsetY: 16,
    },
    plotOptions: {
        pie: {
            donut: {
                size: "65%",
                background: "transparent",
                labels: {
                    show: true,
                    name: {
                        show: true,
                        fontSize: "16px",
                        offsetY: 0,
                        color: "#1c1c1c",
                    },
                    value: {
                        show: true,
                        fontSize: "14px",
                        color: "#1c1c1c",
                        offsetY: 5,
                        formatter: (val) => {
                            return val + "%";
                        },
                    },
                    total: {
                        show: true,
                        label: "Total",
                        color: "#1c1c1c",
                        fontSize: "16px",
                        formatter: (val) => {
                            return 100 + "%";
                        },
                    },
                },
            },
        },
    },
    colors: ["#1C1C1C", "#BAEDBD", "#C6C7F8"],
    labels: ["Competed", "In Progress", "Behind"],
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
var chart4 = new ApexCharts(
    document.querySelector("#projectstatus"),
    projectstatus
);
chart4.render();

//Task overview
var taskoverview = {
    series: [
        {
            name: "This Year",
            data: [25, 38, 35, 29, 32, 28, 25, 32, 11, 18, 27, 30],
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
    colors: ["#A8C5DA"],
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
        tickAmount: 5,
        labels: {
            offsetX: -10,

            offsetY: 0,
            style: {
                fontSize: "12px",
            },
        },
        opposite: false,
    },
    xaxis: {
        categories: [
            "Sprint 1",
            "Sprint 2",
            "Sprint 3",
            "Sprint 4",
            "Sprint 5",
            "Sprint 6",
            "Sprint 7",
            "Sprint 8",
            "Sprint 9",
            "Sprint 10",
            "Sprint 11",
            "Sprint 12",
        ],
        labels: {
            style: {
                fontSize: "12px",
                color: "rgb(55, 61, 63)",
            },
        },
    },
};
var chart5 = new ApexCharts(
    document.querySelector("#taskoverview"),
    taskoverview
);
chart5.render();
