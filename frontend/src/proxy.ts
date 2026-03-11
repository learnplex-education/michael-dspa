import { auth } from "./auth";
import { NextResponse } from "next/server";

export default auth((req) => {
  const email = req.auth?.user?.email;
  const isBerkeleyEmail = email?.endsWith("@berkeley.edu");

  // If they are logged in but don't have a Berkeley email, kick them out
  if (req.auth && !isBerkeleyEmail) {
    return NextResponse.redirect(
      new URL("/api/auth/signin?error=AccessDenied", req.url)
    );
  }
});

export const config = {
  // Protect the home page (chat) and any sub-routes
  matcher: ["/", "/chat/:path*"],
};