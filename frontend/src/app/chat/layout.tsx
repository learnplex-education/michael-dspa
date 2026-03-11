import { redirect } from "next/navigation";

import { auth } from "@/auth";

export default async function ChatLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await auth();
  const email = session?.user?.email ?? "";

  if (!email.endsWith("@berkeley.edu")) {
    redirect("/api/auth/signin?error=AccessDenied");
  }

  return <>{children}</>;
}

