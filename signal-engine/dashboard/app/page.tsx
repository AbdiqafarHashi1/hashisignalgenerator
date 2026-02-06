import dynamic from "next/dynamic";

const LiveDashboard = dynamic(() => import("./components/LiveDashboard"), { ssr: false });

export default function DashboardPage() {
  return <LiveDashboard />;
}
