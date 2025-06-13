/*
-------------------------------------------------------------------------
* Template Name    : Snow - Tailwind CSS Admin & Dashboard Template     * 
* Author           : Webonzer                                           *
* Version          : 1.0.0                                              *
* Created          : March 2023                                         *
* File Description : Main JS file of the template                       *
*------------------------------------------------------------------------
*/
document.addEventListener("alpine:init", () => {
    Alpine.data("sales", () => ({
        init() {
            isDark = this.$store.app.mode === "dark" ? true : false;

            const userchart = null;
            const trafficdevice = null;
            const trafficlocation = null;

            // revenue
            setTimeout(() => {
                this.userchart = new ApexCharts(
                    this.$refs.userchart,
                    this.userchartOptions
                );
                this.$refs.userchart.innerHTML = "";
                this.userchart.render();

                this.trafficdevice = new ApexCharts(
                    this.$refs.trafficdevice,
                    this.trafficdeviceOptions
                );
                this.$refs.trafficdevice.innerHTML = "";
                this.trafficdevice.render();

                this.trafficlocation = new ApexCharts(
                    this.$refs.trafficlocation,
                    this.trafficlocationOptions
                );
                this.$refs.trafficlocation.innerHTML = "";
                this.trafficlocation.render();
            }, 300);

            this.$watch("$store.app.mode", () => {
                isDark = this.$store.app.mode === "dark" ? true : false;

                this.userchart.updateOptions(this.userchartOptions);
                this.trafficdevice.updateOptions(this.trafficdeviceOptions);
                this.trafficlocation.updateOptions(this.trafficlocationOptions);
            });
        },

        get userchartOptions() {
            return {
                chart: {
                    height: 232,
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
                colors: isDark ? ["#fff", "#A8C5DA"] : ["#1c1c1c", "#A8C5DA"],
                markers: {
                    discrete: [
                        {
                            seriesIndex: 0,
                            dataPointIndex: 4,
                            fillColor: "#1C1C1C",
                            strokeColor: "transparent",
                            size: 6,
                        },
                        {
                            seriesIndex: 1,
                            dataPointIndex: 5,
                            fillColor: "#A8C5DA",
                            strokeColor: "transparent",
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
                    tickAmount: 7,
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
                    borderColor: isDark ? "#1c1c1c" : "#e0e6ed",
                    strokeArray: 6,
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
                        opacityFrom: isDark ? 0.19 : 0.28,
                        opacityTo: 0.5,
                        stops: isDark ? [100, 100] : [45, 100],
                    },
                },
            };
        },
        get trafficdeviceOptions() {
            return {
                series: [
                    {
                        name: "Device",
                        data: [10000, 70000, 40000, 50000, 60000, 80000],
                    },
                ],
                chart: {
                    height: 189,
                    type: "bar",
                    events: {
                        click: function (chart, w, e) {},
                    },
                    toolbar: {
                        show: false,
                    },
                },
                colors: [
                    "#BAEDBD",
                    "#C6C7F8",
                    "#1C1C1C",
                    "#B1E3FF",
                    "#95A4FC",
                    "#A1E3CB",
                ],
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
                    categories: [
                        "Linux",
                        "Mac",
                        "iOS",
                        "Windows",
                        "Android",
                        "Other",
                    ],
                    labels: {
                        style: {
                            fontSize: "12px",
                        },
                    },
                },
            };
        },
        get trafficlocationOptions() {
            return {
                series: [38.5, 22.5, 30.8, 8.1],
                chart: {
                    type: "donut",
                    height: 200,
                    fontFamily: "Nunito, sans-serif",
                },
                dataLabels: {
                    enabled: false,
                },
                legend: {
                    position: "right",
                    horizontalAlign: "left",
                    fontSize: "12px",
                    formatter: function (val, opts) {
                        return (
                            val +
                            " - " +
                            opts.w.globals.series[opts.seriesIndex] +
                            "%"
                        );
                    },
                    markers: {
                        width: 6,
                        height: 6,
                        offsetX: -5,
                    },
                    height: 140,
                    offsetY: 0,
                },

                colors: ["#BAEDBD", "#C6C7F8", "#1C1C1C", "#95A4FC"],
                labels: ["United States", "Canada", "Mexico", "Other"],
                responsive: [
                    {
                        breakpoint: 480,
                        options: {
                            legend: {
                                position: "bottom",
                                height: 50,
                            },
                        },
                    },
                ],
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
        },
    }));
});
